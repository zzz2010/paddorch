import paddle.fluid as fluid
from paddle.fluid.framework import   Variable
from paddle.fluid.dygraph.layers import Layer
from paddle.fluid import core

# mypy doesn't understand `with_metaclass` from torch._six
class Function(Layer):  # type: ignore
    def __init__(self , name_scope=None, dtype=core.VarDesc.VarType.FP32):
        super(Function, self).__init__(name_scope,dtype)


        self.saved_tensors=[]

    @classmethod
    def apply(cls, *args,**kwargs):
        function_inst=cls()
        return function_inst.forward(function_inst,*args,**kwargs)

    # @staticmethod
    # def apply( *args):
    #     Function_inst=Function()
    #     return Function_inst.forward(*args)

    def save_for_backward(self,*args):
        for a in args:
            self.saved_tensors.append(a)



def grad(outputs, inputs, grad_outputs=None,retain_graph=False,
        create_graph=True,  only_inputs=True,allow_unused=False):

    return fluid.dygraph.grad(outputs, inputs,grad_outputs=grad_outputs,
         retain_graph=retain_graph,
         create_graph=create_graph,
         only_inputs=only_inputs,
         allow_unused=allow_unused,
         no_grad_vars=None,
         backward_strategy=None)
