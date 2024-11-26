import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
import modules.resnet as resnet
from cbam import CBAM


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs

class CSA(nn.Module):
    def __init__(self, inplanes, planes=1024):
        super(TCNHead, self).__init__()

        self.con1 = nn.Sequential(nn.AvgPool3d(kernel_size=(1,4,4),stride=(1,4,4)),nn.Conv3d(64,512,kernel_size=(1,2,2),stride=(1,2,2)))
        self.con2 = nn.Conv3d(128,512,kernel_size=(1,4,4),stride=(1,4,4))
        self.con3 = nn.Conv3d(256,512,kernel_size=(1,2,2),stride=(1,2,2))
        self.tcn_layers = nn.Sequential(
            nn.Conv1d(inplanes, planes, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, ceil_mode=False),
            nn.Conv1d(planes, planes, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, ceil_mode=False),
        )
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 , 1000)



    def forward(self, res):
        #res_size = [56,28,14,7]
        B,_,T,_,_=res[1].shape
        multi = []

        multi.append(self.con2((res[1])))
        multi.append(self.con3((res[2])))
        multi.append((res[3]))
        for i in range(3):

          multi[i] = multi[i].transpose(1,2).contiguous()
          multi[i] = multi[i].view((-1,)+multi[i].size()[2:])

          multi[i] = self.avgpool(multi[i])
          multi[i] = multi[i].view(multi[i].size(0), -1) 
          multi[i] = self.fc(multi[i])
          multi[i] = multi[i].view(B, T, -1).permute(0,2,1)
          multi[i] = self.tcn_layers(multi[i])

        return multi



class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        #self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d = getattr(resnet, c2d_type)()
        self.conv2d.fc = Identity()
        

        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)

        self.csa = CSA(1000)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        #self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            #inputs = x.reshape(batch * temp, channel, height, width)
            #framewise = self.masked_bn(inputs, len_x)
            #framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
            framewise,res = self.conv2d(x.permute(0,2,1,3,4)) # btc -> bct
            framewise = framewise.view(batch, temp, -1).permute(0,2,1)
        else:
            # frame-wise features
            framewise = x

        res2 = self.csa(res)
        for i in range(3):
            res2[i] = res2[i].permute(2,0,1)
            res2[i] = self.classifier(res2[i])  



        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            #"framewise_features": framewise,
            #"visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
            "res2":              res2,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
                #loss +=  self.loss['CTCLoss'](ret_dict["res2"][0].log_softmax(-1),
                #                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                #                                      label_lgt.cpu().int()).mean()   
                loss +=  self.loss['CTCLoss'](ret_dict["res2"][0].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()    
                loss +=  self.loss['CTCLoss'](ret_dict["res2"][1].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean() 
                loss +=  self.loss['CTCLoss'](ret_dict["res2"][2].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()    
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
                #####loss +=   self.loss['mse_loss'](ret_dict["res1"][0],
                #                                           ret_dict["res1"][1].detach())
                #loss +=   self.loss['mse_loss'](ret_dict["res1"][2],
                              #                                                      ret_dict["res1"][3].detach())
                #loss +=    self.loss['mse_loss'](ret_dict["res1"][4],
                #                                           ret_dict["res1"][5].detach()) 
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        self.loss['mse_loss'] = nn.MSELoss()
        return self.loss