from __future__ import unicode_literals
from torch.autograd import Variable
from io import open
import torch
import torch.nn as nn
import random 
import pickle

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.lin_xh = nn.Linear(input_size, hidden_size)
        self.lin_hh = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.lin_ho = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        self.hidden = self.sigmoid(self.lin_xh(input) + self.lin_hh(hidden))
        self.output = self.lin_ho(self.hidden)
        self.output = self.softmax(self.output)

        return self.output, self.hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

vocab = {}
def encode_word(word):
    vocab_size = len(vocab.keys())
    word_tensor = torch.zeros(1, vocab_size)
    if word not in vocab:
        word_tensor[0][vocab_size - 1] = 1
    else:
        word_tensor[0][vocab[word]] = 1

    return word_tensor

def train(rnn, criterion, learning_rate, input_tensor, target_tensor):
    prev_hidden = rnn.initHidden()

    rnn.zero_grad()
    loss = 0

    for i in range(input_tensor.size()[0]):
        output, prev_hidden = rnn.forward(input_tensor[i], prev_hidden)
        loss += criterion(output, target_tensor[i])

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]

def readData():
    lines = open('input.txt', encoding='utf-8').readlines()
    lines = [line.lower() for line in lines]
    words = ' '.join(lines).split()
    word_dict = {}
    for word in words:
        if word not in word_dict:
            word_dict[word] = 1
        else:
            word_dict[word] += 1

    words_sorted_alphabetically = sorted(word_dict.iteritems(), key = lambda x: x[0])

    for i in range(len(words_sorted_alphabetically)):
        vocab[words_sorted_alphabetically[i][0]] = i

    vocab['<start>'] = i + 1 
    vocab['<end>'] = i + 2

    vocab_size = i + 3

    return lines, vocab_size

def listToTensor(ls):
    return torch.stack(ls)

def lineToTensor(line):
    words = line.split()
    word_tensors = map(encode_word, words)
    input_tensor = listToTensor([encode_word('<start>')] + word_tensors[:len(word_tensors)])
    #target_tensor = listToTensor(word_tensors[1:len(word_tensors)] + [encode_word('<end>')])
    target_tensor = torch.LongTensor([vocab[word] for word in words] + [vocab['<end>']])

    return input_tensor, target_tensor

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(lines):
    line = randomChoice(lines)
    input_line_tensor, target_line_tensor = lineToTensor(line)
    input_line_tensor = Variable(input_line_tensor)
    target_line_tensor = Variable(target_line_tensor)

    return input_line_tensor, target_line_tensor

def main():
    lines, vocab_size = readData()
    with open('vocab.pickle', 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rnn = RNN(vocab_size, 256, vocab_size)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.005

    running_loss = 0
    num_iterations = 10000
    print_every = 50
    for i in range(num_iterations):
        input_tensor, target_tensor = randomTrainingExample(lines)
        output, loss = train(rnn, criterion, learning_rate, input_tensor, target_tensor)

        running_loss += loss

        if i % print_every == 0:
            print('(%d %d%%) %.4f' % (i, i / num_iterations * 100, loss))

    with open('model.pt', 'wb') as f:
        torch.save(rnn, f)


main()
