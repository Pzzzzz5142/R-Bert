import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import torch.nn as nn
import os
from transformers import *


class Trainer(object):
    def __init__(self, model, train_set, test_set):
        self.model = model.cuda()
        self.train_set = train_set
        self.test_set = test_set

    def train(self, tokenizer,  num_train_epochs=5, lr=2e-5, batch_size=16):
        self.model.train()
        train_sampler = RandomSampler(self.train_set)
        train_dataloader = DataLoader(
            self.train_set, shuffle=True, batch_size=batch_size)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr)
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_F = nn.CrossEntropyLoss()
        tj = 0
        tm = 0
        self.model.zero_grad()
        train_iterator = trange(int(num_train_epochs), desc='Epoch')
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            tm = 0
            tj = 0
            for __, batch in enumerate(epoch_iterator):
                outputs = self.model(batch[0], attention_mask=batch[1],
                                     e1_mask=batch[3], e2_mask=batch[4])

                #regular_loss = 0
                '''
                for ___, i in enumerate(self.model.parameters()):
                    regular_loss += torch.sum(
                        abs(i))
                '''

                loss = loss_F(outputs, batch[2])
                tj += float(loss)
                tm += 1
                #loss = loss+regular_loss*0.001
                # print('loss = ',loss)
                self.model.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.evalu('./BERT/Happy')
            print('loss = ', tj/tm)
        # self.save_model()

    def get_class(self, x):
        ls = x.tolist()
        index, val = 0, ls[0]
        for i in range(1, len(ls)):
            if val < ls[i]:
                val = ls[i]
                index = i
        return index

    def evalu(self, path, wh=''):
        self.model.eval()
        se = self.train_set if wh == '1' else self.test_set
        eval_sampler = SequentialSampler(se)
        eval_dataloader = DataLoader(
            se, sampler=eval_sampler, batch_size=16)
        Total = 0
        Right = 0

        labels = open(
            r'D:\OneDrive\VSCode\python\BERT/data/label.txt').read().split('\n')
        file_for_check = open(path+'/data/my_ans.txt'+wh, 'w')
        loss_F = nn.CrossEntropyLoss()
        cnt = 0
        loss = 0
        for _, batch in enumerate(tqdm(eval_dataloader, desc='Eva')):
            with torch.no_grad():
                outputs = self.model(
                    batch[0], attention_mask=batch[1], e1_mask=batch[3], e2_mask=batch[4])

                loss += float(loss_F(outputs, batch[2]))
                cnt += 1

                for i in range(len(outputs)):
                    clstype = self.get_class(outputs[i])
                    if clstype == batch[2].tolist()[i]:
                        Right += 1
                    Total += 1
                    file_for_check.write('%d\t%s\n' %
                                         (Total if wh == '1' else Total+8000, labels[clstype]))
        file_for_check.close()
        os.system(r' perl D:\OneDrive\VSCode\python\BERT\SemEval2010_task8_all_data\SemEval2010_task8_scorer-v1.2\semeval2010_task8_scorer-v1.2.pl ' +
                  path+r'/data/my_ans.txt'+wh+r' D:\OneDrive\VSCode\python\BERT\data\ans_key.txt')
        print('Accuracy = %f %%, total = %d ' % (Right/Total*100, Total))
        print(float(loss/cnt))

    def save_model(self):
        self.model.save_pretrained('./BERT/model')
