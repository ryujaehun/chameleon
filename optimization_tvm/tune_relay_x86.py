
import logging,sys
sys.path.append('/home/jaehun/workspace/chameleon/python')
sys.path.append('/home/jaehun/workspace/chameleon/topi/python')

import os,datetime
import numpy as np
import argparse
import tvm
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner,RLTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from tvm.contrib import util,graph_runtime
import tvm.contrib.graph_runtime as runtime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# logging.basicConfig(level=logging.DEBUG)
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

parser = argparse.ArgumentParser(prog='autotune-llvm')
parser.add_argument('--network', type=str, default='resnet-34',help='choose network')
parser.add_argument('--target', type=str, default='llvm', help='choose optmization target')
parser.add_argument('--batch', type=int, default=4, help='choose batch size')
parser.add_argument('--thread', type=int, default=8, help='choose num of threads')
parser.add_argument('--tuner', type=str, default='rl', help='choose num of threads')# xgb,ga
parser.add_argument('--n_trial', type=int, default=1<<8, help='choose num of threads')
parser.add_argument('--opt_level', type=int, default=2, help='choose opt_level')

opt = parser.parse_args()
print('network ',opt.network)
print('target ',opt.target)
print('batch ',opt.batch)
print('tuner ',opt.tuner)
print('n_trial ',opt.n_trial)

print('opt_level ',opt.opt_level)
#################################################################



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
    elif name == 'squeezenet_v1.0':
        mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.0', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={input_name: input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape


# Replace "llvm" with the correct target of your CPU.
# For example, for AWS EC2 c5 instance with Intel Xeon
# Platinum 8000 series, the target should be "llvm -mcpu=skylake-avx512".
# For AWS EC2 c4 instance with Intel Xeon E5-2666 v3, it should be
# "llvm -mcpu=core-avx2".
target =opt.target
time="_".join(str(datetime.datetime.now()).split())
time=os.path.join('/home/jaehun/workspace/chameleon/optimization_tvm/results/llvm',time)
os.makedirs(time,exist_ok=True)
batch_size = opt.batch#1
dtype = "float32"
model_name =opt.network# "resnet-18"
log_file =time+ "/%s.log" % model_name
graph_opt_sch_file = time+"/%s_graph_opt.log" % model_name
# Set the input name of the graph
# For ONNX models, it is typically "0".
input_name = "data"

# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
num_threads = opt.thread#16
os.environ["TVM_NUM_THREADS"] = str(num_threads)



tuning_option = {
    'log_filename': log_file,
    'tuner': opt.tuner,
    'early_stopping': None,
    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(timeout=10),
    ),
    'n_trial':opt.n_trial,
}


# You can skip the implementation of this function for this tutorial.
def tune_kernels(tasks,
                 measure_option,
                 tuner='gridsearch',
                 early_stopping=None,
                 log_filename=time+'/tuning.log',n_trial=1000):

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # converting conv2d tasks to conv2d_NCHWc tasks
        # op_name = tsk.workload[0]
        # if op_name == 'conv2d':
        #     func_create = 'topi_x86_conv2d_NCHWc'
        # elif op_name == 'dense':
        #     func_create = 'topi_nn_dense'
        # elif op_name == 'depthwise_conv2d_nchw':
        #     func_create = 'topi_x86_depthwise_conv2d_NCHWc_from_nchw'
        # else:
        #     raise ValueError("Tuning {} is not supported on x86".format(op_name))
        #
        #
        # task = autotvm.task.create(func_create, args=tsk.args,
        #                         target=target, template_key='direct')
        # task.workload = tsk.workload


        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        elif tuner == 'rl':
            tuner_obj = RLTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)



        n_trial=n_trial#min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)])


# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if it takes too long to finish.
def tune_graph(graph, dshape, records, opt_sch_file, use_DP=True):
    target_op = [relay.nn.conv2d]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target,log_file=time+"/graph_tuner.log")
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, data_shape, out_shape = get_network(model_name, batch_size)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params, ops=(relay.op.get("nn.conv2d"),))

    # run tuning tasks

    print("Tuning...")

    tune_kernels(tasks, **tuning_opt)
    #tune_graph(mod["main"], data_shape, log_file, graph_opt_sch_file)

    # compile kernels with graph-level best records
    with autotvm.apply_history_best(log_file):
    #with autotvm.apply_graph_best(graph_opt_sch_file):
        print("Compile...")
        with relay.build_config(opt_level=opt.opt_level):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # upload parameters to device

        ctx = tvm.cpu()
        data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
        module = runtime.create(graph, lib, ctx)
        module.set_input(input_name, data_tvm)
        module.set_input(**params)

        # evaluate
        # print("Evaluate inference time cost...")
        # ftimer = module.module.time_evaluator("run", ctx, number=100, repeat=3)
        # prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        # print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
        #       (np.mean(prof_res), np.std(prof_res)))
        save_graph(graph,params,lib)
        #load_graph(ctx=ctx,data_shape=data_shape,out_shape=out_shape)

# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.


def save_graph(graph,params,lib):
    temp = os.path.join(os.getcwd(),time)
    path_lib =os.path.join(temp,"deploy_lib.tar")
    lib.export_library(path_lib)
    with open(os.path.join(temp,"deploy_graph.json"), "w") as fo:
        fo.write(graph)
    with open(os.path.join(temp,"deploy_param.params"), "wb") as fo:
        fo.write(relay.save_param_dict(params))
        print(os.listdir(temp))
# def load_graph(ctx,data_shape,out_shape,path=None):
#     if path is None:
#         path=os.path.join(os.getcwd(),time)

#     path_lib = os.path.join(path,"deploy_lib.tar")
#     loaded_json = open(os.path.join(path,"deploy_graph.json")).read()
#     loaded_lib = tvm.runtime.load_module(path_lib)
#     loaded_params = bytearray(open(os.path.join(path,"deploy_param.params"), "rb").read())
#     input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))

#     module = graph_runtime.create(loaded_json, loaded_lib, ctx)
#     module.load_params(loaded_params)
#     module.run(data=input_data)
#     out_deploy = module.get_output(0).asnumpy()

#     # Print first 10 elements of output
#     print(out_deploy.flatten()[0:10])
#     # check whether the output from deployed module is consistent with original one
#     out = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()
#     tvm.testing.assert_allclose(out_deploy, out, atol=1e-3)

tune_and_evaluate(tuning_option)
