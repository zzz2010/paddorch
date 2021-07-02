import paddorch as torch
class TorchBenchmarkBase(torch.nn.Module):
    """ This is a base class used to create Pytorch operator benchmark.
        module_name is the name of the operator being benchmarked.
        test_name is the name (it's created by concatenating all the
        inputs) of a specific test
    """

    def __init__(self):
        super(TorchBenchmarkBase, self).__init__()
        self.user_given_name = None
        self._pass_count = 0
        self._num_inputs_require_grads = 0

    def _set_backward_test(self, is_backward):
        self._is_backward = is_backward

    def auto_set(self):
        """ This is used to automatically set the require_grad for the backward patch.
            It is implemented based on two counters. One counter to save the number of
            times init has been called. The other counter to save the number of times
            this function itself has been called. In the very first time init is called,
            this function counts how many inputs require gradient. In each of the
            following init calls, this function will return only one true value.
            Here is an example:
                ...
                self.v1 = torch.rand(M, N, K, requires_grad=self.auto_set())
                self.v2 = torch.rand(M, N, K, requires_grad=self.auto_set())
                ...
        """
        if not self._is_backward:
            return False

        if self._pass_count == 0:
            self._num_inputs_require_grads += 1
            return True
        else:
            self._auto_set_counter += 1
            return (self._pass_count == self._auto_set_counter)

    def extract_inputs_tuple(self):
        self.inputs_tuple = tuple(self.inputs.values())


    def get_inputs(self):
        # Need to convert the inputs to tuple outside of JIT so that
        # JIT can infer the size of the inputs.
        return self.inputs_tuple


    def forward_impl(self):
        # This is to supply the inputs to the forward function which
        # will be called in both the eager and JIT mode of local runs
        return self.forward(*self.get_inputs())


    def forward_consume(self, iters: int):
        #  _consume is used to avoid the dead-code-elimination optimization
        for _ in range(iters):
            torch.ops.operator_benchmark._consume(self.forward_impl())

    def module_name(self):
        """ this is used to label the operator being benchmarked
        """
        if self.user_given_name:
            return self.user_given_name
        return self.__class__.__name__

    def set_module_name(self, name):
        self.user_given_name = name

    def test_name(self, **kargs):
        """ this is a globally unique name which can be used to
            label a specific test
        """

        # This is a list of attributes which will not be included
        # in the test name.
        skip_key_list = ['device']

        test_name_str = []
        for key in kargs:
            value = kargs[key]
            test_name_str.append(
                ('' if key in skip_key_list else key)
                + str(value if type(value) != bool else int(value)))
        name = (self.module_name() + '_' +
                '_'.join(test_name_str)).replace(" ", "")
        return name


