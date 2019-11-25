import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import torch.nn as nn

from transformers import *


class Trainer(object):
    def __init__(self, model):
        self.model = model.cuda()

    def train(self, train_dataset, tokenizer, dropoutRate, num_train_epochs=5, lr=2e-5, batch_size=16):
        self.model.train()
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=batch_size)
        optimizer = AdamW(self.model.parameters(), lr=lr)
        loss_F = nn.CrossEntropyLoss()
        train_iterator = trange(int(num_train_epochs), desc='Epoch')
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                outputs = self.model(batch[0], attention_mask=batch[1],
                                     e1_mask=batch[3], e2_mask=batch[4])

                loss = loss_F(outputs, batch[2])
                #print('loss = ',loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # self.save_model()

    def get_class(self, x):
        ls = x.tolist()
        index, val = 0, ls[0]
        for i in range(1, len(ls)):
            if val < ls[i]:
                val = ls[i]
                index = i
        return index

    def evalu(self, path, test_dataset):
        self.model.eval()
        eval_sampler = SequentialSampler(test_dataset)
        eval_dataloader = DataLoader(
            test_dataset, sampler=eval_sampler, batch_size=16)
        Total = 0
        Right = 0

        labels = open(path+'/label.txt').read().split('\n')
        file_for_check = open(path+'/my_ans.txt', 'w')

        for batch in tqdm(eval_dataloader, desc='Eva'):
            with torch.no_grad():
                outputs = self.model(
                    batch[0], attention_mask=batch[1], e1_mask=batch[3], e2_mask=batch[4])

                for i in range(len(outputs)):
                    clstype = self.get_class(outputs[i])
                    if clstype == batch[2].tolist()[i]:
                        Right += 1
                    Total += 1
                    file_for_check.write('%d\t%s\n' %
                                         (Total+8000, labels[clstype]))

        print('Accuracy = %f %%, total = %d ' % (Right/Total*100, Total))

    def save_model(self):
        self.model.save_pretrained('./BERT/model')
