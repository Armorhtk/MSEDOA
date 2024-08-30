import torch
import torch.nn as nn
import torch.nn.functional as F

class DOADNN(nn.Module):
  def __init__(self, input_dim, output_dim, L, hidden_size=[256, 1024, 2048, 1024, 512]):
    super(DOADNN, self).__init__()
    self.input_dim = input_dim * 2 * L
    layers = []
    self.hidden_size = [self.input_dim] + hidden_size
    for input_size, output_size in zip(self.hidden_size[:-1], self.hidden_size[1:]):
      layers.append(nn.Linear(input_size, output_size))
      layers.append(nn.ReLU())
    layers.append(nn.Linear(self.hidden_size[-1], output_dim))
    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    bts, M, L = x.shape
    x = torch.view_as_real(x)
    x = x.permute(0, 1, 3, 2)
    x = x.contiguous().view(bts, M * 2 * L) 
    out = self.layers(x)
    return out
