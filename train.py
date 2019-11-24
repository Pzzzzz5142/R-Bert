import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm, trange
import torch.nn as nn

from transformers import *


def train(train_dataset, model, tokenizer, num_train_epochs, lr=2e-5, batch_size=16):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=batch_size)
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_F = nn.CrossEntropyLoss()
    train_iterator = trange(int(num_train_epochs), desc='Epoch')
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            inputs = torch.tensor([i.input_ids for i in batch]).cuda()
            labels = torch.tensor([i.label_id for i in batch]).cuda()
            attention_mask = torch.tensor(
                [i.attention_mask for i in batch]).cuda()
            outputs = model(inputs, attention_mask=attention_mask)

            loss = loss_F(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
