import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune
from models.resnet import ResNet18
from pruning import *


device = torch.device("cuda")
loaded_model = ResNet18()
checkpoint = torch.load("./checkpoint/ckpt.pth")
loaded_model = torch.nn.DataParallel(loaded_model)
loaded_model.load_state_dict(checkpoint['net'])
# prune.random_unstructured(loaded_model.module.linear, name = "weight", amount = 0.5)
# print(list(loaded_model.module.linear.named_parameters()))
# for name, layer in loaded_model.named_modules():
#     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
#         print(name, layer.weight.size())
module = loaded_model.module
iterative_pruning(module, 0.5, 0.05)
print(check_global_sparsity(module))