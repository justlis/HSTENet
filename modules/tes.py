import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalExtensionShift(nn.Module):
    def __init__(self,in_channels, n_div=8, inplace=False):
        super(TemporalExtensionShift, self).__init__()


        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')

        self.in_channels = in_channels
        self.fold = self.in_channels // n_div
        self.exten_shift = nn.Conv1d(
                                    self.in_channels, self.in_channels,
                                    kernel_size=3, padding=3, groups=self.in_channels,dilation=3,
                                    bias=False)  

        self.exten_shift.weight.requires_grad = True

        self.exten_shift.weight.data.zero_()
        self.exten_shift.weight.data[:self.fold, 0, 2] = 1 # shift left
        self.exten_shift.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right  
        if 2*self.fold < self.in_channels:
            self.exten_shift.weight.data[2 * self.fold:, 0, 1] = 1 # fixed




    def forward(self,x):
        n,c, t, h, w = x.size()
        x_shift = x.permute([0, 3, 4, 1, 2]) 
        x_shift = x_shift.contiguous().view(n*h*w, c, t)      
        x_shift = self.exten_shift(x_shift)  # (n_batch*h*w, c, n_segment)      
        x_shift = x_shift.view(n, h, w, c, t)
        x_shift = x_shift.permute([0, 3, 4, 1, 2]) 

    
        return x_shift
