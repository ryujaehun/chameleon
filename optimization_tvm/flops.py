import os,datetime
import numpy as np
import argparse
import tvm
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
from tvm.contrib import util,graph_runtime
import tvm.contrib.graph_runtime as runtime
import argparse,subprocess
parser = argparse.ArgumentParser(prog='inference-autotune')
parser.add_argument('--path', type=str, default=1, help='choose batch size')
parser.add_argument('--batch', type=int, default=1, help='choose batch size')
opt = parser.parse_args()
data_shape=(opt.batch, 3, 224, 224)
out_shape=(opt.batch, 1000)

def load_graph(ctx,data_shape,out_shape=(1, 1000),path=None):
    if path is None:
        path=os.path.join(os.getcwd(),'time')

    path_lib = os.path.join(path,"deploy_lib.tar")
    loaded_json = open(os.path.join(path,"deploy_graph.json")).read()
    loaded_lib = tvm.runtime.load_module(path_lib)
    loaded_params = bytearray(open(os.path.join(path,"deploy_param.params"), "rb").read())
    input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))

    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)
    module.run(data=input_data)
    #The function will be invoked (1 + number x repeat) times, with the first call discarded in case there is lazy initialization.
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, number=100, repeat=3)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
            (np.mean(prof_res), np.std(prof_res)))
    out_deploy = module.get_output(0).asnumpy()
    out = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()
    tvm.testing.assert_allclose(out_deploy, out, atol=1e-3)
load_graph(tvm.cpu(),data_shape,out_shape,opt.path)
