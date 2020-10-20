import torch.optim as optim
from layers.scheduler_base import SchedulerBase

class SGD(SchedulerBase):
  def __init__(self, model):
    super(SGD, self).__init__()
    self.model = model
    self._lr = 0.01
    self._optimizer = optim.SGD(model.parameters(), self._lr, momentum=0.9, weight_decay=1e-5)

  def schedule(self, epoch, epochs, **kwargs):
    lr = 0.01
    for param_group in self._optimizer.param_groups:
      param_group['lr'] = lr
    self._lr = self._optimizer.param_groups[0]['lr']
    return self._optimizer, self._lr

class SGD2a(SchedulerBase):
  def __init__(self, model):
    super(SGD2a, self).__init__()
    self.model = model
    self._lr = 0.005
    self._optimizer = optim.SGD(model.parameters(), self._lr, momentum=0.9, weight_decay=1e-5)

  def schedule(self, epoch, epochs, **kwargs):
    lr = 0.005
    for param_group in self._optimizer.param_groups:
      param_group['lr'] = lr
    self._lr = self._optimizer.param_groups[0]['lr']
    return self._optimizer, self._lr

class SGD2c(SchedulerBase):
  def __init__(self, model):
    super(SGD2c, self).__init__()
    self.model = model
    self._lr = 0.0025
    self._optimizer = optim.SGD(model.parameters(), self._lr, momentum=0.9, weight_decay=1e-5)

  def schedule(self, epoch, epochs, **kwargs):
    lr = 0.0025
    for param_group in self._optimizer.param_groups:
      param_group['lr'] = lr
    self._lr = self._optimizer.param_groups[0]['lr']
    return self._optimizer, self._lr

