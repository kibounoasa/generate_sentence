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
import matplotlib.pyplot as plt

class LSTM_net(nn.Module):
    def __init__(self, corpus_max):
        super(LSTM_net, self).__init__()
        self.embed = nn.Embedding(corpus_max, int(corpus_max/3))
        self.lstm = nn.LSTM(input_size=int(corpus_max/3), hidden_size=int(corpus_max/6), batch_first=True)
        self.out = nn.Linear(int(corpus_max/6), corpus_max)
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
    max_seq_len =  16#逆伝搬で切り取る文字の長さ指定
    batch_size = 32

    loss_plot = []
    while True:
        seq_len = 256 #学習で切り取る長さ
        train_data, label_data = util.make_batch(corpus, seq_len, batch_size)
        train_data = torch.tensor(train_data, dtype=torch.int64)
        label_data = torch.tensor(label_data, dtype=torch.int64)
        
        loss_epoch = 0    
        count = 0
        for t in range(0, seq_len, max_seq_len):
            train_seq_batch = train_data[:, t:t+max_seq_len]
            label_seq_batch = label_data[:, t:t+max_seq_len]
            out = model(x=train_seq_batch, t_bptt=True)
            loss = loss_function.softmax_cross_entropy(out, label_seq_batch[:, -1])
            opt.zero_grad()
            loss.backward()
            opt.step()    
            loss_epoch += loss.data.numpy()
            count += 1
        loss_epoch /= count
        epoch += 1
        sys.stdout.write( "\r|| epoch : " + str(epoch) + " || loss : " + str(loss_epoch) + " ||")
        
        model.init_hidden_cell()
        
        loss_plot.append(loss_epoch)
        if epoch % 10 == 0:
            np.save("loss", np.array(loss_epoch))
            plt.cla()
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.plot(loss_plot, label="loss")
            plt.savefig("loss.png")


        #generate_sentence_test
        if gen_flag:
            print("\n=================generate=====================\n")
            #test
            index = random.randint(0, corpus.max()-1)
            gen_sentence = [index]
            print(convert.id2char[index], end="")
            for c in range(700):
                now_input = np.array(gen_sentence)
                now_input = torch.tensor(now_input[np.newaxis], dtype=torch.int64)
                out = F.softmax(model(now_input, t_bptt=True), dim=1).data.numpy()[0]
                next_index = int(np.random.choice(len(out), size=1, p=out)[0]) #ランダムにサンプリング
                #next_index = int(out.argmax()) #最大確率をサンプリング
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