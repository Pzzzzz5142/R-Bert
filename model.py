from transformers import *
import torch
from torch import torch.nn as nn


class MyBert(nn):
    def __init__(self):
        super(MyBert, self).__init__()

        self.brt = BertModel().from_pretrained('bert-base-uncased')
        