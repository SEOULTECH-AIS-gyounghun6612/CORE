# ----- INFORMATION ----- #
# first edit date   : 2020/06/02
# first editer      : choi keonghun
# last edit date    : 2020/08/26
# last editer       : choi keonghun


# --------------- PROGRAM SETTING --------------- #
# Import module
import math
import torch
import torch.distributions as distributions

import torch.cuda as torch_cuda

# Import custom moudule

# Set constant
PI = math.pi
SMALL_VALUE = 1e-5

# setting
def is_cuda():
    return torch.cuda.is_available()


# dataloader
def np_to_tensor(np_array):
    return torch.FloatTensor(np_array)


# layers
def Flatten(input):
    return input.view(input.size(0), -1)

class Unflatten(torch.nn.Module):
    def __init__(self, shape):
        super(Unflatten, self).__init__()
        self.m_shape = shape

    def forward(self, input):
        return input.view(input.size(0), self.m_shape[0], self.m_shape[1], self.m_shape[2]) 

def custom_cuncat(tensors, dim=0):
    return torch.cat(tensors, dim)

# model
def save(epoch, weight, save_file_dir):
    save_dic = {'epoch': epoch,
                'model_state_dict': weight}

    torch.save(save_dic, save_file_dir)

def load(model, data_path):
    checkpoint = torch.load(data_path)
    model.load_state_dict(checkpoint['model_state_dict'])

def make_optim(paramaters, lr):
    return torch.optim.Adam(paramaters, lr)

def make_dist(mu, std):
        # esp = torch.randn(*mu.size()).cuda() if is_cuda else torch.randn(*mu.size())
        # return mu + std * esp
        return distributions.Normal(mu, std)

def make_WdS_padding(S, K):
    if (K - S) % 2 == 0:
        return (K - S) / 2, (K - S) / 2
    else:
        return (K - S) // 2, (K - S) - ((K - S) // 2)
