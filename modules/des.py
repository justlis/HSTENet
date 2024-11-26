import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalExtensionShift(nn.Module):
    def __init__(self,in_channels, n_div=8, inplace=False):
        super(TemporalExtensionShift, self).__init__()
         #self.net = net
        #self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        #Eprint('=> Using fold div: {}'.format(self.fold_div))
        self.in_channels = in_channels
        self.fold = self.in_channels // n_div
        self.action_shift = nn.Conv1d(
                                    self.in_channels, self.in_channels,
                                    kernel_size=3, padding=2, groups=self.in_channels,dilation=2,
                                    bias=False)  

        self.action_shift.weight.requires_grad = True

        self.action_shift.weight.data.zero_()
        self.action_shift.weight.data[:self.fold, 0, 2] = 1 # shift left
        self.action_shift.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right  
        if 2*self.fold < self.in_channels:
            self.action_shift.weight.data[2 * self.fold:, 0, 1] = 1 # fixed


    #def forward(self, x):
    #    x = self.shift(x, fold_div=self.fold_div, inplace=self.inplace)
    #    return x
# self.alpha[0]
    #@staticmethod
    def forward(self,x):
        n,c, t, h, w = x.size()
         #n_batch = nt // n_segment
         #x = x.view(n_batch, n_segment, c, h, w)
        x_shift = x.permute([0, 3, 4, 1, 2]) 
        #print(x_shift.shape)
        x_shift = x_shift.contiguous().view(n*h*w, c, t)      
        x_shift = self.action_shift(x_shift)  # (n_batch*h*w, c, n_segment)      
        x_shift = x_shift.view(n, h, w, c, t)
        x_shift = x_shift.permute([0, 3, 4, 1, 2]) 

    
        return x_shift
#x = torch.randn(2,512,20,7,7)
#ts = TemporalShift(512,8)
#y = ts(x)
#print(y.shape)
