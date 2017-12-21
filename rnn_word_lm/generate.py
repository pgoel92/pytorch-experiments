###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable

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

def invert_vocab(vocab):
    inverted_vocab = {}
    for key, value in vocab.iteritems():
        inverted_vocab[value] = key

    return inverted_vocab

def encode_word(word, vocab):
    vocab_size = len(vocab.keys())
    word_tensor = torch.zeros(1, vocab_size)
    if word not in vocab:
        word_tensor[0][vocab_size - 1] = 1
    else:
        word_tensor[0][vocab[word]] = 1

    return word_tensor

with open('model.pt', 'rb') as f:
    model = torch.load(f)

with open('vocab.pickle', 'rb') as handle:
    vocab = pickle.load(handle)

inverted_vocab = invert_vocab(vocab)
vocab_size = len(vocab.keys())
hidden = model.initHidden()
start_tensor = torch.zeros(1, vocab_size)
start_tensor[0][vocab['<end>']] = 1
input = Variable(start_tensor)

output_words = []
max_len = 51
for i in range(max_len):
    output, hidden = model(input, hidden)
    topv, topi = output.data.topk(1)
    topi = topi[0][0]
    word = inverted_vocab[topi]
    output_words.append(word)

    if word == '<end>':
        break
    input = Variable(encode_word(word, vocab))
    # word_weights = output.squeeze()
    # word_idx = torch.multinomial(word_weights, 1)[0]
    # input.data.fill_(word_idx)
    # word = corpus.dictionary.idx2word[word_idx]

    # outf.write(word + ('\n' if i % 20 == 19 else ' '))

    # if i % args.log_interval == 0:
    #     print('| Generated {}/{} words'.format(i, args.words))

print ' '.join(output_words)
