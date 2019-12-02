from train import Trainer
from model import MyBert
from DataLoader import load_datas
from transformers import *
import torch
import numpy as np

if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    net = MyBert(128, 19, dropoutRate=0.1)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<e1>', '<e2>', '</e1>', '</e2>']}) 
    train_dataset = load_datas('./BERT', tokenizer, 128)
    test_dataset = load_datas('./BERT', tokenizer, 128, mode=False)

    train = Trainer(net,train_set=train_dataset,test_set=test_dataset)
    train.train(tokenizer, num_train_epochs=10)
    train.evalu('./BERT', wh='1')
    train.evalu('./BERT')