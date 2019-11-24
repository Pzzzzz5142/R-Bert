from transformers import *
import torch
import torch.nn as nn



class MyBert(nn.Module):
    def __init__(self):
        super(MyBert, self).__init__()

        self.brt = BertModel.from_pretrained('bert-base-uncased')
        self.hidden1=nn.Linear(128,784)