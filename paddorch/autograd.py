import paddle.fluid as fluid
import paddle.static.Variable as Variable


def grad(outputs, inputs, grad_outputs=None,retain_graph=False,
        create_graph=True,  only_inputs=True,allow_unused=False):

    return fluid.dygraph.grad(outputs, inputs,grad_outputs=grad_outputs,
         retain_graph=retain_graph,
         create_graph=create_graph,
         only_inputs=only_inputs,
         allow_unused=allow_unused,
         no_grad_vars=None,
         backward_strategy=None)
