import torch


class OptimizerWrapper(torch.optim.Optimizer):
    r"""
    A wrapper for pytorch.optimizer that forwards all methods to the wrapped optimizer
    """

    def __init__(self, optim: torch.optim.Optimizer):
        object.__init__(self)
        self.optim = optim

    @property
    def defaults(self):
        return self.optim.defaults

    @property
    def state(self):
        return self.optim.state

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.optim)})"

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        return self.optim.load_state_dict(state_dict)

    def step(self, *args, **kwargs):
        return self.optim.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        return self.optim.zero_grad(*args, **kwargs)

    @property
    def param_groups(self):
        return self.optim.param_groups

    def add_param_group(self, param_group: dict) -> None:
        return self.optim.add_param_group(param_group)
