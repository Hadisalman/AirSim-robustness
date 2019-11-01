from abc import ABCMeta, abstractmethod

from IPython import embed
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional

class Attacker(metaclass=ABCMeta):
    @abstractmethod
    def attack(self, inputs, targets, normalize_input):
        raise NotImplementedError

class PGD(Attacker):
    def __init__(self, random_start=True, step_size=0.1, 
                epsilon=0.3, num_steps=40, 
                norm='linf', est_grad=None):
        super(PGD, self).__init__()
        self.criterion = CrossEntropyLoss(reduction='none')
        self.epsilon = epsilon
        self.norm = norm
        self.num_steps = num_steps
        self.random_start = random_start
        self.step_size = step_size
        self.est_grad = est_grad

    def _compute_loss(self, inputs, targets, normalize_input=None):
        if normalize_input is not None:
            logits = self.model(normalize_input(inputs))
        else:
            logits = self.model(inputs)
        loss = self.criterion(logits, targets)
        return loss

    def _perturb_linf(self, inputs, targets, normalize_input=None):
        x = inputs.clone().detach()
        if self.random_start:
            x += torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
            x = torch.clamp(x, 0, 1)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                losses = self._compute_loss(x, targets, normalize_input)
                loss = losses.mean()
            if self.est_grad is None:
                grad = torch.autograd.grad(loss, x)[0]
            else:
                f = lambda _x, _y: self._compute_loss(_x, _y, normalize_input)
                grad = calc_est_grad(f, x, targets, *self.est_grad)
            x = x + self.step_size * torch.sign(grad)
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon).clamp(0, 1)
        return x.detach()


    def _perturb_l2(self, inputs, targets, normalize_input=None):
        batch_size = inputs.shape[0]
        x = inputs.clone().detach()
        if self.random_start:
            x = x + (torch.rand_like(x) - 0.5).renorm(p=2, dim=0, maxnorm=self.epsilon)
            x = torch.clamp(x, 0, 1)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                losses = self._compute_loss(x, targets, normalize_input)
                loss = losses.mean()
            if self.est_grad is None:
                grad = torch.autograd.grad(loss, x)[0]
            else:
                f = lambda _x, _y: self._compute_loss(_x, _y, normalize_input)
                grad = calc_est_grad(f, x, targets, *self.est_grad)
            grad_norms = grad.view(batch_size, -1).norm(p=2, dim=1)
            grad.div_(grad_norms.view(-1, 1, 1, 1))
          
            x = x + self.step_size*grad

            # project on the l2 ball
            x = inputs + torch.renorm(x - inputs, p=2, dim=0, maxnorm=self.epsilon)
            x = torch.clamp(x, 0, 1)
        return x.detach()


    def attack(self, model, inputs, targets, normalize_input=None):
        self.model = model
        if self.norm == 'l2':
            return self._perturb_l2(inputs, targets, normalize_input)        
        elif self.norm == 'linf':
            return self._perturb_linf(inputs, targets, normalize_input) 
        else:
            raise Exception

# Taken from https://github.com/MadryLab/robustness
def calc_est_grad(func, x, y, rad, num_samples):
    B, *_ = x.shape
    Q = num_samples//2
    N = len(x.shape) - 1
    with torch.no_grad():
        # Q * B * C * H * W
        extender = [1]*N
        queries = x.repeat(Q, *extender)
        noise = torch.randn_like(queries)
        norm = noise.view(B*Q, -1).norm(dim=-1).view(B*Q, *extender)
        noise = noise / norm
        noise = torch.cat([-noise, noise])
        queries = torch.cat([queries, queries])
        y_shape = [1] * (len(y.shape) - 1)
        l = func(queries + rad * noise, y.repeat(2*Q, *y_shape)).view(-1, *extender) 
        grad = (l.view(2*Q, B, *extender) * noise.view(2*Q, B, *noise.shape[1:])).mean(dim=0)
    return grad

# Taken from https://github.com/Hadisalman/smoothing-adversarial
class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.
      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means)/sds