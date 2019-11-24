from transformers import *
import torch
import torch.nn as nn


class MyBert(nn.Module):
    def __init__(self, maxLen, classNum):
        super(MyBert, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hidden1 = nn.Linear(784, 784)
        self.hidden2 = nn.Linear(784, 784)
        self.hidden3 = nn.Linear(3*784, classNum)
        self.maxLen = maxLen

    def getaverage(self, hs):

        return res

    def forward(self, datas, attention_mask, e1_mask, e2_mask):
        x, cls_vec = self.bert(datas.input_ids, attention_mask=attention_mask)
        entity1 = torch.tensor([]).cuda()
        entity1 = torch.tensor([]).cuda()
        for i in range(datas):
            tmp1 = x[i][e1_mask[i][0]+1]
            tmp2 = x[i][e2_mask[i][0]+1]
            for j in range(e1_mask[i][0]+2, e1_mask[i][1]):
                tmp1 += x[i][j]
            for j in range(e2_mask[i][0]+2, e2_mask[i][1]):
                tmp2 += x[i][j]
            entity1 = torch.cat(
                (entity1, tmp1/(e1_mask[i][1]-e1_mask[i][0]-1)), 0)
            entity2 = torch.cat(
                (entity2, tmp2/(e2_mask[i][1]-e2_mask[i][0]-1)), 0)
        '''   
        entity1 = torch.tensor([[data[i] for i in range(
            data.e1_mask[0]+1, data.e2_mask[1])] for data in x]).cuda()#[b,j-i+1,len]
        entity2 = torch.tensor([[data[i] for i in range(
            data.e1_mask[0]+1, data.e2_mask[1])] for data in x]).cuda()
        '''
        entity1 = torch.tanh(entity1)
        entity2 = torch.tanh(entity2)
        entity1 = self.hidden1(entity1)
        entity2 = self.hidden1(entity2)
        cls_vec = torch.tanh(cls_vec)
        cls_vec = self.hidden2(cls_vec)
        hidden_vec = torch.cat((cls_vec, entity1, entity2), dim=-1)
        x = self.hidden3(hidden_vec)
        x = torch.softmax(x)
        return x
