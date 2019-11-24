import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm, trange
import torch.nn as nn

from transformers import *


class Trainer(object):
    def __init__(self, model):
        self.model = model

    def train(self, train_dataset, tokenizer, dropoutRate, num_train_epochs=5, lr=2e-5, batch_size=16, device=torch.device('cuda')):
        self.model.to(device)
        self.model.half()
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

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def get_class(self, x):
        ls = x.tolist()
        index, val = 0, ls[0]
        for i in range(1, len(ls)):
            if val < ls[i]:
                val = ls[i]
                index = i
        return index

    def evalu(self, test_dataset, device='cuda'):
        if device=='cpu':
            self.model.float()
        device = torch.device(device)
        self.model.to(device)
        self.model.eval()
        test_dataset = test_dataset.tensors
        Total = len(test_dataset)
        Right = 0

        labels = test_dataset[2].tolist()

        predict = self.model(
            test_dataset[0], attention_mask=test_dataset[1], e1_mask=test_dataset[3], e2_mask=test_dataset[4])
        for i in range(len(predict)):
            if self.get_class(predict[i]) == labels[i]:
                Right += 1

        print('Accuracy = %f %%, total = %d ' % (Right/Total*100, Total))
