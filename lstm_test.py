import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random
import threading

class LSTM_net(nn.Module):
    def __init__(self, corpus_max):
        super(LSTM_net, self).__init__()
        self.embed = nn.Embedding(corpus_max, int(corpus_max/2))
        self.lstm = nn.LSTM(input_size=int(corpus_max/2), hidden_size=int(corpus_max/3), batch_first=True)
        self.out = nn.Linear(int(corpus_max/3), corpus_max)
        self.hidden_cell = None

    def init_hidden_cell(self):
        self.hidden_cell = None

    def repackage_hidden(self, hidden_cell):
        self.hidden_cell = (hidden_cell[0].detach(), hidden_cell[1].detach())

    def forward(self, x, t_bptt=False):
        h = self.embed(x)
        all_time_hidden, hidden_cell = self.lstm(h, self.hidden_cell)
        if t_bptt:
            self.repackage_hidden(hidden_cell)
        out = self.out(all_time_hidden[:, -1, :])
       
        return out

class Loss_function(nn.Module):
    def __init__(self):
        super(Loss_function, self).__init__()
        self.softmax_cross_entropy = nn.CrossEntropyLoss()
        self.mean_squared_error = nn.MSELoss()
        self.softmax = nn.Softmax()

class Convert_char():
    def __init__(self):
        self.char2id = {}
        self.id2char = {}
        self.corpus = None
        self.corpus_onehot = None

    def make_char_id(self, sentence):
        for char in sentence:
            if char not in self.char2id:
                new_id = len(self.char2id)
                self.char2id[char] = new_id
                self.id2char[new_id] = char
        return self.char2id, self.id2char
    
    def convert_corpus(self, sentence):
        char2id, id2char = self.make_char_id(sentence)
        self.corpus = np.array([char2id[c] for c in sentence])
        #self.corpus_onehot = np.eye(N=self.corpus.max()+1)[self.corpus]
        return self.corpus

class Util():   
    def get_sentence(self, txt_path, start=False, end=False):
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        sentence = ""
        if start:
            sentence += "^" #SOSの追加
        for line in lines:
            if(line != "\n"):
                line = line.strip()
                sentence += line
        if end:
            sentence += "_" #EOSの追加
        return sentence

    def make_batch(self, corpus, seq_len, batchsize=5):
        train_data = []
        label_data = []
        for i in range(batchsize):
            start = random.randint(0, len(corpus)-seq_len-1)
            train_batch = corpus[start:start+seq_len]
            label_batch = corpus[start+1:start+seq_len+1]

            train_data.append(train_batch)
            label_data.append(label_batch)

        train_data = np.array(train_data)
        label_data = np.array(label_data)
        return train_data, label_data

    def print_loss(self, data):
        sys.stdout.write("\r" + data)


def main():
    global gen_flag
    txt_path = "./test.txt"

    util = Util()
    convert = Convert_char()

    sentence = util.get_sentence(txt_path, start=True, end=True)
    corpus = convert.convert_corpus(sentence)
    print(convert.char2id)
    
    model = LSTM_net(corpus_max=corpus.max()+1)
    opt = optim.Adam(model.parameters())
    loss_function = Loss_function()
    

    epoch = 0
    random_batch_size = 64
    max_batch_seq_len = 10

    while True:
        seq_len = random.randint(400, 700) #学習に記憶する長さの指定
        seq_len = 600
        train_data, label_data = util.make_batch(corpus, seq_len, random_batch_size)
        train_data = torch.tensor(train_data, dtype=torch.int64)
        label_data = torch.tensor(label_data, dtype=torch.int64)

        if seq_len <= max_batch_seq_len: #hiddenの値を残さない場合
            out = model(x=train_data)
            #loss_t = 0
            #for t in range(out.shape[1]):
            #    loss_t += loss_function.softmax_cross_entropy(out[:, t, :], label_data[: ,t])
            #loss = loss_t / out.shape[1]
            loss = loss_function.softmax_cross_entropy(out, label_data[:, -1])
            opt.zero_grad()
            loss.backward()
            opt.step()    
            util.print_loss(data = "|| loss : " + str(loss.data.numpy()) + " || epoch : " + str(epoch) + " || seq_len : " + str(seq_len) + " || Truncated BPTT : False ||")
        
        else: #hiddenの値を残してTruncated BPTTを行う場合(mondaigaarukanousei)
            for t in range(0, seq_len, max_batch_seq_len):
                train_seq_batch = train_data[:, t:t+max_batch_seq_len]
                label_seq_batch = label_data[:, t:t+max_batch_seq_len]
                out = model(x=train_seq_batch, t_bptt=True)
                #loss_t = 0
                #for k in range(out.shape[1]):
                #    loss_t += loss_function.softmax_cross_entropy(out[:, k, :], label_seq_batch[: ,k])
                #loss = loss_t / out.shape[1]
                loss = loss_function.softmax_cross_entropy(out, label_seq_batch[:, -1]) #ここが間違っていた
                opt.zero_grad()
                loss.backward()
                opt.step()    
                util.print_loss(data = "|| loss : " + str(loss.data.numpy()) + " || epoch : " + str(epoch) + " || seq_len : " + str(seq_len) + " || Truncated BPTT : True ||")
        model.init_hidden_cell()
        epoch += 1

        #generate_sentence_test
        if gen_flag:
            print("\n=================generate=====================\n")
            #test
            index = random.randint(0, corpus.max()-1)
            #index = 0
            gen_sentence = [index]
            print(convert.id2char[index], end="")
            for c in range(700):
                now_input = np.array(gen_sentence)
                now_input = torch.tensor(now_input[np.newaxis], dtype=torch.int64)
                #out = F.softmax(model(now_input), dim=2).data.numpy()[0, 0]
                out = F.softmax(model(now_input, t_bptt=True), dim=1).data.numpy()[0]
                next_index = int(np.random.choice(len(out), size=1, p=out)[0]) #ランダムにサンプリング
                #next_index = int(out.argmax()) #最大確率をサンプリング
                #print("argmax:", out.argmax(), "sample_index", next_index)
                print(convert.id2char[next_index], end="")
                if((c+2)%100==0):
                    print("")
                gen_sentence = [next_index]
            gen_flag = False
            print("\n\n================================================")
            model.init_hidden_cell()
        

def generate_flag():
    global gen_flag
    gen_flag = False
    while True:
        key = input()
        gen_flag = True

if __name__ == "__main__":
    thread = threading.Thread(target=generate_flag)
    thread.daemon = True
    thread.start()
    main()