from transformers import *
import torch
import torch.nn as nn


class MyBert(nn.Module):
    def __init__(self, maxLen, classNum, dropoutRate):
        super(MyBert, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hidden1 = nn.Linear(768, 768)
        nn.init.xavier_uniform_(self.hidden1.weight)
        self.hidden2 = nn.Linear(768, 768)
        nn.init.xavier_uniform_(self.hidden2.weight)
        self.hidden3 = nn.Linear(3*768, classNum)
        nn.init.xavier_uniform_(self.hidden3.weight)
        self.tanh=nn.Tanh()
        self.dropout = nn.Dropout(dropoutRate)
        self.maxLen = maxLen

    def forward(self, datas, attention_mask, e1_mask, e2_mask):
        x, y = self.bert(datas, attention_mask=attention_mask)
        y = self.dropout(y)
        cls_vec = None
        entity1 = None
        entity2 = None
        flg = True
        for i in range(len(datas)):
            tmp1 = x[i][e1_mask[i][0]+1]
            tmp2 = x[i][e2_mask[i][0]+1]
            assert int(e1_mask[i][0]) < int(e1_mask[i][1])
            assert int(e2_mask[i][0]) < int(e2_mask[i][1])
            for j in range(int(e1_mask[i][0]+2), int(e1_mask[i][1])):
                tmp1 = tmp1 + x[i][j]
            for j in range(int(e2_mask[i][0]+2), int(e2_mask[i][1])):
                tmp2 = tmp2 + x[i][j]
            if flg:
                entity1 = (
                    tmp1/float(e1_mask[i][1]-e1_mask[i][0]-1)).unsqueeze(0)
                entity2 = (
                    tmp2/float(e1_mask[i][1]-e1_mask[i][0]-1)).unsqueeze(0)
                cls_vec = x[i][0].unsqueeze(0)
                flg = False
            else:
                entity1 = torch.cat(
                    (entity1, (tmp1/float(e1_mask[i][1]-e1_mask[i][0]-1)).unsqueeze(0)), dim=0)
                entity2 = torch.cat(
                    (entity2, (tmp2/float(e2_mask[i][1]-e2_mask[i][0]-1)).unsqueeze(0)), dim=0)
                cls_vec = torch.cat((cls_vec, x[i][0].unsqueeze(0)), dim=0)
        '''
        entity1 = torch.tensor([[data[i] for i in range(
            data.e1_mask[0]+1, data.e2_mask[1])] for data in x]).cuda()#[b,j-i+1,len]
        entity2 = torch.tensor([[data[i] for i in range(
            data.e1_mask[0]+1, data.e2_mask[1])] for data in x]).cuda()
        '''
        cls_vec = self.dropout(cls_vec)
        cls_vec = self.tanh(cls_vec)
        cls_vec = y
        cls_vec = self.hidden1(cls_vec)
        entity1 = self.dropout(entity1)
        entity2 = self.dropout(entity2)
        entity1 = self.tanh(entity1)
        entity2 = self.tanh(entity2)
        entity1 = self.hidden2(entity1)
        entity2 = self.hidden2(entity2)
        hidden_vec = self.cat((cls_vec, entity1, entity2), dim=-1)
        hidden_vec = self.dropout(hidden_vec)
        x = self.hidden3(hidden_vec)
        return x
