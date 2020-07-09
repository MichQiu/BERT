import numpy as np
import torch.optim as op

class ScheduledOptim(op.lr_scheduler):
    """ Wrapper for learning rate scheduling"""

    super().__init__()

    def __init__(self, optimizer, model_d, n_warmup_steps):
        """
        Args:
            optimizer: Optimization algorithm
            model_d: model dimension
            n_warmup_steps: number of warmup steps
        """
        self._optimizer = optimizer
        self.model_d = model_d
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0 # number of training steps

    def step_and_update_lr(self):
        """ Step with the inner optimizer """
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """Zero out the gradients with the inner optimizer"""
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        model_d = self.model_d
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        # Increase learning rate linearly for the warmup steps then decrease it inversely proportional to n_steps
        return (model_d ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """

        self.n_steps += 1
        lr = self._get_lr_scale()

        # Update learning rate in optimizer's param_groups dict
        for param_group in self._optimizer.param_groups():
            param_group['lr'] = lr


