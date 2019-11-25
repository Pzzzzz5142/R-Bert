from transformers import *
import torch
import torch.nn as nn


class MyBert(nn.Module):
    def __init__(self, maxLen, classNum, dropoutRate):
        super(MyBert, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hidden1 = nn.Linear(768, 768)
        nn.init.xavier_normal(self.hidden1.weight)
        self.hidden2 = nn.Linear(768, 768)
        nn.init.xavier_normal(self.hidden2.weight)
        self.hidden3 = nn.Linear(3*768, classNum)
        nn.init.xavier_normal(self.hidden3.weight)
        self.dropout = nn.Dropout(dropoutRate)
        self.dropout1 = nn.Dropout(dropoutRate)
        self.dropout2 = nn.Dropout(dropoutRate)
        self.dropout3 = nn.Dropout(dropoutRate)

        self.maxLen = maxLen

    def getaverage(self, hs):

        return

    def forward(self, datas, attention_mask, e1_mask, e2_mask):
        x, cls_vec = self.bert(datas, attention_mask=attention_mask)
        entity1 = None
        entity2 = None
        flg = True
        for i in range(len(datas)):
            tmp1 = x[i][e1_mask[i][0]+1]
            tmp2 = x[i][e2_mask[i][0]+1]
            for j in range(int(e1_mask[i][0]+2), int(e1_mask[i][1])):
                tmp1 = tmp1 + x[i][j]
            for j in range(int(e2_mask[i][0]+2), int(e2_mask[i][1])):
                tmp2 = tmp2 + x[i][j]
            if flg:
                entity1 = tmp1.unsqueeze(0)
                entity2 = tmp2.unsqueeze(0)
                flg = False
            else:
                entity1 = torch.cat(
                    (entity1, (tmp1/float(e1_mask[i][1]-e1_mask[i][0]-1)).unsqueeze(0)), dim=0)
                entity2 = torch.cat(
                    (entity2, (tmp2/float(e2_mask[i][1]-e2_mask[i][0]-1)).unsqueeze(0)), dim=0)
        '''   
        entity1 = torch.tensor([[data[i] for i in range(
            data.e1_mask[0]+1, data.e2_mask[1])] for data in x]).cuda()#[b,j-i+1,len]
        entity2 = torch.tensor([[data[i] for i in range(
            data.e1_mask[0]+1, data.e2_mask[1])] for data in x]).cuda()
        '''
        entity1 = torch.tanh(entity1)
        entity2 = torch.tanh(entity2)
        entity1 = self.dropout(entity1)
        entity2 = self.dropout1(entity2)
        entity1 = self.hidden1(entity1)
        entity2 = self.hidden1(entity2)
        hidden_vec = torch.cat((cls_vec, entity1, entity2), dim=-1)
        hidden_vec = self.dropout3(hidden_vec)
        x = self.hidden3(hidden_vec)
        #x = torch.softmax(x, dim=1)
        return x
