from transformers import *
import torch
import torch.nn as nn



class MyBert(nn.Module):
    def __init__(self,maxLen,classNum):
        super(MyBert, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hidden1=nn.Linear(784,784)
        self.hidden2=nn.Linear(784,784)
        self.hidden3=nn.Linear(3*784,classNum)
        self.maxLen=maxLen

    def getaverage(self,hs):
        return

    def forward(self,datas):
        
        self.bert(datas)


        return