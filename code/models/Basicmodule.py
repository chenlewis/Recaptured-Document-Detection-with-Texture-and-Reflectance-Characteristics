import torch.nn as nn
import torch as t
import time

class Basicmodule(nn.Module):

    def __init__(self):
        super(Basicmodule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_deacy):
        return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_deacy)

class Flat(nn.Module):
    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)