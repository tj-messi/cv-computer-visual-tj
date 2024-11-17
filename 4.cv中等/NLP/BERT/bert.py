import math
import re
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer

#gelu激活函数,相比与relu在零点处梯度更为平滑
def gelu(x):
    return 0.5 * x * (1+torch.tanh(math.sqrt(2/math.pi)*(x+0.044715*torch.pow(x, 3))))


