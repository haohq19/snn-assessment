from models.sresnet import *
from models.scnn import *
from snn_evaluate import Model
from fvcore.nn import *

import torch
import torch.nn as nn

#  统计模型参数数量、计算量
def parm_statics(
        model,
        device
        ):
    
    flops = FlopCountAnalysis(model.to(device), torch.randn(1, 2, 32, 32, 10).to(device))
    print(parameter_count_table(model))
    print(flops.total()/1024/1024/1024)
    
    if __name__ == '__main__':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = Model(device=device, n_class=100, m_name='sres5')
        parm_statics(model, device)



