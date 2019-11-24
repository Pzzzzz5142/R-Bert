import csv
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import TensorDataset


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(self, input_ids, attention_mask, label_id, e1_mask, e2_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        #self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask


def load_datas(path, tokenizer, max_len, pad_token=0, add_special_token=True, mode=True):
    features = []
    labels = open(path+'/label.txt').read().split('\n')
    with open(path+('/train.tsv' if mode else '/test.tsv')) as open_file:
        for line in open_file.readlines():
            line = line.split('\t')
            label = line[0]
            text = line[1]
            tokened = tokenizer.tokenize(
                ('[CLS] ' if add_special_token else '') + text)
            e1_b = tokened.index('<e1>')
            e1_e = tokened.index('</e1>')
            e2_b = tokened.index('<e2>')
            e2_e = tokened.index('</e2>')
            tokened[e1_b] = '$'
            tokened[e1_e] = '$'
            tokened[e2_b] = '#'
            tokened[e2_e] = '#'
            input_ids = tokenizer.convert_tokens_to_ids(tokened)
            attention_mask = [1 if pad_token == 0 else 1]*len(input_ids)
            padding_length = max_len-len(input_ids)
            attention_mask += [pad_token]*padding_length
            input_ids += [pad_token]*padding_length
            assert len(input_ids) == max_len
            assert len(attention_mask) == max_len
            features.append(InputFeatures(input_ids, attention_mask, label_id=labels.index(label),
                                          e1_mask=(e1_b, e1_e), e2_mask=(e2_b, e2_e)))

    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long)
    all_e1_mask = torch.tensor(
        [f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
    all_e2_mask = torch.tensor(
        [f.e2_mask for f in features], dtype=torch.long)  # add e2 mask

    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_label_ids, all_e1_mask, all_e2_mask)

    return dataset


if __name__ == '__main__':
    import torch
    import transformers
    token = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    token.add_special_tokens({'additional_special_tokens': [
                             '<e1>', '<e2>', '</e1>', '</e2>']})
    load_datas('./BERT/data', max_len=128, tokenizer=token)
