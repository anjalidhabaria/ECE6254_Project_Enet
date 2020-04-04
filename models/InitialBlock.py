# Reproducing the paper                                          
# ENet - Fast Scene Understanding for Autonomous Driving                         
# Paper: https://arxiv.org/pdf/1708.02550.pdf                   
# Credits: Following class code is borrowed from iArunava/ENet-Real-Time-Semantic-Segmentation                             
##################################################################

import torch
import torch.nn

class InitialBlock(nn.Module):
  
    # Initial block of the model:
    #         Input
    #        /     \
    #       /       \
    #maxpool2d    conv2d-3x3
    #       \       /  
    #        \     /
    #      concatenate
   
    def __init__ (self,in_channels = 3,out_channels = 13):
        super().__init__()


        self.maxpool = nn.MaxPool2d(kernel_size=2, 
                                      stride = 2, 
                                      padding = 0)

        self.conv = nn.Conv2d(in_channels, 
                                out_channels,
                                kernel_size = 3,
                                stride = 2, 
                                padding = 1)

        self.prelu = nn.PReLU(16)

        self.batchnorm = nn.BatchNorm2d(out_channels)
  
    def forward(self, x):
        
        main = self.conv(x)
        main = self.batchnorm(main)
        
        side = self.maxpool(x)
        
        # concatenating on the channels axis
        x = torch.cat((main, side), dim=1)
        x = self.prelu(x)
        
        return x
