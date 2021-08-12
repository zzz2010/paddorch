import paddle.fluid as fluid
from paddle.fluid.framework import   Variable
from paddle.fluid.dygraph.layers import Layer
from paddle.fluid import core
from collections import defaultdict
import paddle
# mypy doesn't understand `with_metaclass` from torch._six
class Function(Layer):  # type: ignore
    '''
    the backward function of this class CAN NOT declare as staticmethod, remove @staticmethod
    '''
    def __init__(self , name_scope=None, dtype=core.VarDesc.VarType.FP32):
        super(Function, self).__init__(name_scope,dtype)
        self.saved_tensors=[]
        self.grad_cache=defaultdict(lambda :0)
        self.hook_helper=dict()

    def register_hook(self,var,name):
        def set_grad(grad):
            if name   in self.grad_cache:
                if grad is None:
                    grad=0
                grad.set_value(self.grad_cache[name]+grad)
            else:
                print(name,"NOT found in grad_cache")
            return grad
        helper=var.register_hook(set_grad)
        self.hook_helper[name]=helper
    def delete_hook(self,name):
        if name in self.hook_helper:
            self.hook_helper[name].remove()
            del self.grad_cache[name]

    @classmethod
    def apply(cls, *args,**kwargs):
        """ different to pytorch, the forward method can not be defined as static, because need to store some grad cache"""
        function_inst=cls()
        return function_inst.forward(  *args,**kwargs)

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
