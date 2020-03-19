import numpy as np
import tvm
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
from tvm.contrib import util,graph_runtime
import tvm.contrib.graph_runtime as runtime
import argparse
parser = argparse.ArgumentParser(prog='naive_inference')
parser.add_argument('--network', type=str, default='resnet-34',help='choose network')
parser.add_argument('--batch', type=int, default=4, help='choose batch size')
parser.add_argument('--opt_level', type=int, default=0, help='choose opt_level')
opt = parser.parse_args()


dtype='float32'
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
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape



mod, params, data_shape, out_shape = get_network(opt.network, opt.batch)



with relay.build_config(opt_level=opt.opt_level):
    graph, lib, params = relay.build_module.build(
        mod, target='llvm -mcpu=core-avx2', params=params)
input_name = "data"
print('network ',opt.network)
print('batch ',opt.batch)
print('opt_level ',opt.opt_level)

# upload parameters to device

ctx = tvm.cpu()
data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype('float32'))
module = runtime.create(graph, lib, ctx)
module.set_input(input_name, data_tvm)
module.set_input(**params)

# evaluate

print("Evaluate inference time cost...")
ftimer = module.module.time_evaluator("run", ctx, number=100, repeat=3)
prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
    (np.mean(prof_res), np.std(prof_res)))