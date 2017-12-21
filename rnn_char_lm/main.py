from __future__ import unicode_literals, print_function, division
from torch.autograd import Variable
from io import open
import torch
import torch.nn as nn
import glob
import unicodedata
import string
import time
import math
import random

all_letters = string.ascii_letters + " .,;'()#?"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    line = open(filename, encoding='utf-8').read().strip()
    line = '#'.join(line.split('\n'))
    i = 0
    lines = []
    while i < len(line):
        eos = min(i + 25, len(line))
        lines.append(line[i:eos])
        i += 20
    return lines
    #return [unicodeToAscii(line) for line in lines]

def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def inputTensor(line):
    if len(line) > 1:
        line = line[:-1]
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    return torch.LongTensor(letter_indexes)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.lin_xh = nn.Linear(input_size, hidden_size)
        self.lin_hh = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.lin_ho = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        hidden = self.tanh(self.lin_xh(input) + self.lin_hh(hidden))
        output = self.lin_ho(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

n_hidden = 512 

#def tensorToLetter(tensor):
#    pass

#tensor = Variable(lineToTensor("Prateek"));
#hidden = Variable(torch.zeros(1, n_hidden))
#output, hidden = rnn.forward(tensor[0], hidden) 

criterion = nn.NLLLoss()
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(rnn, input_line_tensor, target_line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()
    loss = 0

    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn(input_line_tensor[i], hidden)
        loss += criterion(output, target_line_tensor[i])

    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]

# Sample from a category and starting letter
def sample(model, sentence_length, start_letter='S'):
    input = Variable(inputTensor(start_letter))
    hidden = model.initHidden()

    output_name = start_letter

    for i in range(sentence_length):
        output, hidden = model(input[0], hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0]
        if topi == n_letters - 2:
            #break
            letter = all_letters[topi]
            output_name += '\n'
        else:
            letter = all_letters[topi]
            output_name += letter
        input = Variable(inputTensor(letter))

    return output_name

# Get multiple samples from one category and multiple starting letters
def samples(model, start_letter='J'):
    #for start_letter in start_letters:
    print(sample(model, 300, start_letter))
#rnn = RNN(n_letters, 128, n_letters)

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(current_index, lines):
    #line = randomChoice(lines)
    line = lines[current_index]
    input_line_tensor = Variable(inputTensor(line))
    target_line_tensor = Variable(targetTensor(line))
    return input_line_tensor, target_line_tensor

def main():
    try:
        f = open('model.pt', 'rb')
        rnn = torch.load(f)
        samples(rnn)
    except:
        rnn = RNN(n_letters, n_hidden, n_letters)
    
        lines = readLines("input.txt")
        print(lines)
        # Random item from a list
        
        n_iters = 5000
        print_every = 100
        plot_every = 5
        all_losses = []
        total_loss = 0 # Reset every plot_every iters
        
        start = time.time()
       
        current_index = 0
        for iter in range(1, n_iters + 1):
            output, loss = train(rnn, *randomTrainingExample(current_index, lines))
            current_index += 1
            if current_index == len(lines):
                current_index = 0
            total_loss += loss
        
            if iter % print_every == 0:
                print('(%d %d%%) %.4f' % (iter, iter / n_iters * 100, loss))
      
            if iter % plot_every == 0:
                all_losses.append(total_loss / plot_every)
                total_loss = 0

            if loss <= 0.1:
                break
        
        with open('model.pt', 'wb') as f:
            torch.save(rnn, f)
   
main()
