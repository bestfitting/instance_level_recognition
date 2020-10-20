import torch.optim as optim
import torch.nn as nn
class SchedulerBase(object):
    def __init__(self):
        self._is_load_best_weight = True
        self._is_load_best_optim = True
        self._is_adjust_lr = True
        self._lr = 0.01
        self._optimizer = None

    def schedule(self,net, epoch, epochs, **kwargs):
        raise Exception('Did not implemented')

    def is_load_best_weight(self):
        return self._is_load_best_weight

    def is_load_best_optim(self):
        return self._is_load_best_optim


    def reset(self):
        self._is_load_best_weight = True
        self._load_best_optim = True


    def is_adjust_lr(self):
        return self._is_adjust_lr

    def get_optimizer(self):
        return self._optimizer