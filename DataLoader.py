import csv
from tqdm import tqdm
import numpy as np
import os


class DataLoader(object):
    def __init__(self, path, readNum):
        self.readNum = readNum
        self.path = path

    def load(self):
        datas, labels = self.__read__()
        for i in range(len(datas)):
            for j in range(len(datas[i])):
                if datas[i][j] == '<':
                    if datas[i][j+2] == '1':
                        datas[i] = datas[i][:j]+' $  '+datas[i][j+4:]
                    elif datas[i][j+2] == '2':
                        datas[i] = datas[i][:j]+' #  '+datas[i][j+4:]
                    elif datas[i][j+3] == '1':
                        datas[i] = datas[i][:j]+' $   '+datas[i][j+5:]
                    elif datas[i][j+3] == '2':
                        datas[i] = datas[i][:j]+' $   '+datas[i][j+5:]
                        break
        for i in range(len(labels)):
            for j in range(len(labels)):
                if labels[i][j] == '(':
                    labels[i] = labels[i][:j]
                    break
        return datas, labels

    def __read__(self):
        with open(self.path) as fl:
            reader = csv.reader(fl, delimiter='\t')
            cnt = 0
            datas = []
            labels = []
            for row in reader:
                if cnt == 0:
                    datas.append(row[1])
                elif cnt == 1:
                    labels.append(row[0])
                cnt += 1
                if cnt == 4:
                    cnt = 0

        return datas, labels


if __name__ == "__main__":
    a = 'list(range<e1>(10))122112'
    for i in range(len(a)):
        if a[i] == '<':
            a = a[:i]+' & '+a[i+4:]
            break

    print(a)
