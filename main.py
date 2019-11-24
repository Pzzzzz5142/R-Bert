from train import Trainer
from model import MyBert
from DataLoader import load_datas
from transformers import *

if __name__ == '__main__':
    net = MyBert(128, 19, dropoutRate=0.1).cuda()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<e1>', '<e2>', '</e1>', '</e2>']})
    train_dataset = load_datas('./BERT/data', tokenizer, 128)
    test_dataset = load_datas('./BERT/data', tokenizer, 128, mode=False)

    train = Trainer(net)
    train.train(train_dataset, tokenizer, dropoutRate=0.1,num_train_epochs=5)
    train.evalu(test_dataset)
