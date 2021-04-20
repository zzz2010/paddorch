import paddle.fluid as fluid
from paddle.fluid.framework import   Variable
from paddle.fluid.dygraph.layers import Layer


# mypy doesn't understand `with_metaclass` from torch._six
class Function(Layer):  # type: ignore
    r"""Records operation history and defines formulas for differentiating ops.

    See the Note on extending the autograd engine for more details on how to use
    this class: https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd

    Every operation performed on :class:`Tensor` s creates a new function
    object, that performs the computation, and records that it happened.
    The history is retained in the form of a DAG of functions, with edges
    denoting data dependencies (``input <- output``). Then, when backward is
    called, the graph is processed in the topological ordering, by calling
    :func:`backward` methods of each :class:`Function` object, and passing
    returned gradients on to next :class:`Function` s.

    Normally, the only way users interact with functions is by creating
    subclasses and defining new operations. This is a recommended way of
    extending torch.autograd.

    Examples::

        >>> class Exp(Function):
        >>>
        >>>     @staticmethod
        >>>     def forward(ctx, i):
        >>>         result = i.exp()
        >>>         ctx.save_for_backward(result)
        >>>         return result
        >>>
        >>>     @staticmethod
        >>>     def backward(ctx, grad_output):
        >>>         result, = ctx.saved_tensors
        >>>         return grad_output * result
        >>>
        >>> #Use it by calling the apply method:
        >>> output = Exp.apply(input)
    """

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "Legacy autograd function with non-static forward method is deprecated. "
            "Please use new-style autograd function with static forward method. "
            "(Example: https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function)")

    # for the tracer
    is_traceable = False


    @staticmethod
    def forward(ctx , *args , **kwargs )  :
        r"""Performs the operation.

        This function is to be overridden by all subclasses.

        It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).

        The context can be used to store tensors that can be then retrieved
        during the backward pass.
        """
        raise NotImplementedError("You must implement the forward function for custom"
                                  " autograd.Function.")

    @staticmethod
    def backward(ctx , *grad_outputs )  :
        r"""Defines a formula for differentiating the operation.

        This function is to be overridden by all subclasses.

        It must accept a context :attr:`ctx` as the first argument, followed by
        as many outputs did :func:`forward` return, and it should return as many
        tensors, as there were inputs to :func:`forward`. Each argument is the
        gradient w.r.t the given output, and each returned value should be the
        gradient w.r.t. the corresponding input.

        The context can be used to retrieve tensors saved during the forward
        pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
        of booleans representing whether each input needs gradient. E.g.,
        :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
        first input to :func:`forward` needs gradient computated w.r.t. the
        output.
        """
        raise NotImplementedError("You must implement the backward function for custom"
                                  " autograd.Function.")


def grad(outputs, inputs, grad_outputs=None,retain_graph=False,
        create_graph=True,  only_inputs=True,allow_unused=False):

    return fluid.dygraph.grad(outputs, inputs,grad_outputs=grad_outputs,
         retain_graph=retain_graph,
         create_graph=create_graph,
         only_inputs=only_inputs,
         allow_unused=allow_unused,
         no_grad_vars=None,
         backward_strategy=None)
