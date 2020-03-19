# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Auto-tuning a convolutional network for NVIDIA GPU
==================================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, `Eddie Yan <https://github.com/eqy/>`_

Auto-tuning for specific devices and workloads is critical for getting the
best performance. This is a tutorial on how to tune a whole convolutional
network for NVIDIA GPU.

The operator implementation for NVIDIA GPU in TVM is written in template form.
The template has many tunable knobs (tile factor, unrolling, etc).
We will tune all convolution and depthwise convolution operators
in the neural network. After tuning, we produce a log file which stores
the best knob values for all required operators. When the TVM compiler compiles
these operators, it will query this log file to get the best knob values.

We also released pre-tuned parameters for some NVIDIA GPUs. You can go to
`NVIDIA GPU Benchmark <https://github.com/apache/incubator-tvm/wiki/Benchmark#nvidia-gpu>`_
to see the results.
"""

######################################################################
# Install dependencies
# --------------------
# To use the autotvm package in tvm, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost tornado
#
# To make TVM run faster during tuning, it is recommended to use cython
# as FFI of tvm. In the root directory of tvm, execute:
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.

import os,datetime
import argparse
import numpy as np

import tvm
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from tvm.contrib import util,graph_runtime
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime

parser = argparse.ArgumentParser(prog='autotune-cuda')
parser.add_argument('--network', type=str, default='resnet-34', help='choose network')
parser.add_argument('--batch', type=int, default=1, help='choose batch size')
parser.add_argument('--tuner', type=str, default='random', help='choose num of threads')
parser.add_argument('--n_trial', type=int, default=2, help='choose num of threads')
parser.add_argument('--time', type=str, default='200', help='choose num of threads')
opt = parser.parse_args()
#################################################################
# Define Network
# --------------
# First we need to define the network in relay frontend API.
# We can load some pre-defined network from :code:`tvm.relay.testing`.
# We can also load models from MXNet, ONNX and TensorFlow.

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet':
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'squeezenet_v1.1':
        mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape

###########################################
# Set Tuning Options
# ------------------
# Before tuning, we apply some configurations.

#### DEVICE CONFIG ####
target = tvm.target.cuda()
# time="_".join(str(datetime.datetime.now()).split())
# time=os.path.join('results/cuda',time)
# os.makedirs(time)
time=opt.time

#### TUNING OPTION ####
model_name =opt.network
log_file =time+ "/%s.log" % model_name
graph_opt_sch_file = time+"/%s_graph_opt.log" % model_name
dtype = 'float32'
input_name = "data"
tuning_option = {
    'log_filename': log_file,
    'tuner': opt.tuner,
    'n_trial': opt.n_trial,
    'early_stopping': 1000,
    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150)

     ),
}

## ISSUE : https://discuss.tvm.ai/t/autotvm-localrunner-not-working-on-windows/969/3


def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename=time+'/tuning.log',
               use_transfer_learning=True,
               try_winograd=True):
    if try_winograd:
        for i in range(len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception:
                pass
    
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=tsk_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)
                       ])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.
def tune_graph(graph, dshape, records, opt_sch_file, use_DP=True):
    target_op = [relay.nn.conv2d]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target,log_file=time+"/graph_tuner.log")
    executor.benchmark_layout_transform(min_exec_num=2000,target_host='cuda')
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)

def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape, out_shape = get_network(model_name, batch_size=1)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params, ops=(relay.op.nn.conv2d,))

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)
    #tune_graph(mod["main"], input_shape, log_file, graph_opt_sch_file)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # export library

        filename = os.path.join(os.getcwd(),time)+"/net.tar"
        lib.export_library(filename)

        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.create(graph, lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        # print("Evaluate inference time cost...")
        # ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
        # prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        # print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
        #       (np.mean(prof_res), np.std(prof_res)))
        save_graph(graph,params,lib)
        # load_graph(ctx=ctx,data_shape=input_shape,out_shape=out_shape)
def save_graph(graph,params,lib):
    temp = os.path.join(os.getcwd(),time)
    path_lib =os.path.join(temp,"deploy_lib.tar")
    lib.export_library(path_lib)
    with open(os.path.join(temp,"deploy_graph.json"), "w") as fo:
        fo.write(graph)
    with open(os.path.join(temp,"deploy_param.params"), "wb") as fo:
        fo.write(relay.save_param_dict(params))
        print(os.listdir(temp))
def load_graph(ctx,data_shape,out_shape,path=None):
    if path is None:
        path=os.path.join(os.getcwd(),time)

    path_lib = os.path.join(path,"deploy_lib.tar")
    loaded_json = open(os.path.join(path,"deploy_graph.json")).read()
    loaded_lib = tvm.runtime.load_module(path_lib)
    loaded_params = bytearray(open(os.path.join(path,"deploy_param.params"), "rb").read())
    input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))

    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)
    module.run(data=input_data)
    out_deploy = module.get_output(0).asnumpy()
    # Print first 10 elements of output
    print(out_deploy.flatten()[0:10])
    # check whether the output from deployed module is consistent with original one
    out = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()
    tvm.testing.assert_allclose(out_deploy, out, atol=1e-3)
# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

tune_and_evaluate(tuning_option)

