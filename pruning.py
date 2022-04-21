from tabnanny import check
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune

def one_shot_pruning(module, sparsity):
    for name, layer in module.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            #do stuff here, will be updated very soon
            prune.random_unstructured(layer, name="weight", amount = sparsity)


def iterative_pruning(module, target_sparsity, delta, x):
    curr_sparsity = check_global_sparsity(module)
    while abs(curr_sparsity - target_sparsity) > delta:
        one_shot_pruning(module, x)
        curr_sparsity = check_global_sparsity(module)


        
#Iterates through the 
def check_global_sparsity(module):
    num = 0
    denom = 0
    for name, layer in module.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            num += torch.sum(layer.weight == 0)
            denom += layer.weight.nelement()
    return num/denom


