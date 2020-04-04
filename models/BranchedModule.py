# Reproducing the paper                                          
# ENet - Fast Scene Understanding for Autonomous Driving                         
# Paper: https://arxiv.org/pdf/1708.02550.pdf                                            
##################################################################

from .ENetDecoder import ENetDecoder

class BranchedModule(nn.Module):
    def __init__(self, C):
        super(BranchedModule, self).__init__()
        self.C = C
        self.layer1 = ENetBranch(C)
        self.layer2 = ENetBranch(C)
        self.layer3 = ENetBranch(C)
        
    def forward(self,x):
        y = [self.layer1(x), self.layer2(x), self.layer3(x)]
        return y

