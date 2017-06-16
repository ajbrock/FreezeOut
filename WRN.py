## Wide ResNet with FreezeOut
# Based on code by xternalz: https://github.com/xternalz/WideResNet-pytorch
# WRN by Sergey Zagoruyko and Nikos Komodakis
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from utils import scale_fn
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate,layer_index):
        super(BasicBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
                               
        # If the layer is being trained or not
        self.active = True
        
        # The layer index relative to the rest of the net
        self.layer_index = layer_index
        
    def forward(self, x):
    
        if not self.active:
            self.eval()
            
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        out = torch.add(x if self.equalInOut else self.convShortcut(x), out)
        if self.active:
            return out
        else:
            return out.detach()
    
    



# note: we call it DenseNet for simple compatibility with the training code.
# similar we call it growthRate instead of widen_factor
class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, nClasses, epochs, t_0, scale_lr=True, how_scale = 'cubic', const_time=False, dropRate=0.0):
        super(DenseNet, self).__init__()
        
        widen_factor=growthRate
        num_classes = nClasses
        self.epochs = epochs
        self.t_0 = t_0
        self.scale_lr = scale_lr
        self.how_scale = how_scale
        self.const_time = const_time
        
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = int((depth - 4) / 6)
        # print(type(n))
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv1.layer_index = 0
        self.conv1.active = True
        self.layer_index = 1
        
        # 1st block
        self.block1 = self._make_layer(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = self._make_layer(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = self._make_layer(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        
        self.bn1.active=True
        self.fc.active=True
        self.bn1.layer_index = self.layer_index
        self.fc.layer_index = self.layer_index

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
            
            if hasattr(m,'active'):
                m.lr_ratio = scale_fn[self.how_scale](self.t_0 + (1 - self.t_0) * float(m.layer_index) / self.layer_index)
                m.max_j = self.epochs * 1000 * m.lr_ratio
                
                # Optionally scale the learning rates to have the same total
                # distance traveled (modulo the gradients).
                m.lr = 1e-1 / m.lr_ratio if self.scale_lr else 1e-1
                
        # Optimizer
        self.optim = optim.SGD([{'params':m.parameters(), 'lr':m.lr, 'layer_index':m.layer_index} for m in self.modules() if hasattr(m,'active')],  
                         nesterov=True,momentum=0.9, weight_decay=1e-4)
        # Iteration Counter            
        self.j = 0  

        # A simple dummy variable that indicates we are using an iteration-wise
        # annealing scheme as opposed to epoch-wise. 
        self.lr_sched = {'itr':0}
    def _make_layer(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        layers = []
        print(nb_layers,type(nb_layers))
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate,self.layer_index))
            self.layer_index +=1
        return nn.Sequential(*layers)
    
    def update_lr(self):
    
        # Loop over all modules
        for m in self.modules():
        
            # If a module is active:
            if hasattr(m,'active') and m.active:
            
                # If we've passed this layer's freezing point, deactivate it.
                if self.j > m.max_j: 
                    m.active = False
                    
                    # Also make sure we remove all this layer from the optimizer
                    for i,group in enumerate(self.optim.param_groups):
                        if group['layer_index']==m.layer_index:
                            self.optim.param_groups.remove(group)
                
                # If not, update the LR
                else:
                    for i,group in enumerate(self.optim.param_groups):
                        if group['layer_index']==m.layer_index:
                            self.optim.param_groups[i]['lr'] = (0.05/m.lr_ratio)*(1+np.cos(np.pi*self.j/m.max_j))\
                                                              if self.scale_lr else 0.05 * (1+np.cos(np.pi*self.j/m.max_j))
        self.j += 1   
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return F.log_softmax(self.fc(out))