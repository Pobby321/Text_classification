import math
import random
def load_data(file_path):
    f = open(file_path,'r',encoding='utf-8')
    target = []
    sentences = []
    for line in f.readlines():
        label, sentence = line.split('\t')
        target.append(label)
        sentences.append(sentence)
    f.close()
    return sentences,target

def prepare_dataset(data,char_to_id,tag_to_id):
    dataset = []
    for i in range(len(data)):
        string = [char_to_id.get(w,10000) for w in data[0][i]]
        target = [tag_to_id.get(t,10000) for t in data[1][i]]
        dataset.append([string,target])
    return dataset

class BatchManager():
    def __init__(self,data,batch_size):
        self.batch_size = self.sort_and_pad(data,batch_size)
        self.len_data = len(self.batch_size)

    def sort_and_pad(self,data,batch_size):
        num_batch = int(math.ceil(len(data)/batch_size))
        sorted_data = sorted(data,key=lambda x:len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*int(batch_size):(i+1)*int(batch_size)]))
        return batch_data

    def pad_data(self,data):
        strings = []
        targets = []
        max_len = max([len(sentence[0]) for sentence in data])

        for line in data:
            string, target = line
            padding_len = [0]*max_len-len(string)
            strings.append(string+padding_len)
            targets.append(target)
        return [strings,targets]

    def batch_iter(self,shuffle=True):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]