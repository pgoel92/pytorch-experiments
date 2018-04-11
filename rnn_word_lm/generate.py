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

import model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prefix')
args = parser.parse_args()

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

def decode_word(output, inverted_vocab):
    topv, topi = output.data.topk(1)
    topi = topi[0][0]
    word = inverted_vocab[topi]
    return word

def input_tensor_from_list(ls, vocab, vocab_size):
    nwords = len(ls)
    input_tensor = torch.zeros(nwords, 1, vocab_size) 
    for i in range(len(ls)):
        inp = encode_word(ls[i], vocab)
        input_tensor[i] = inp

    return Variable(input_tensor)

with open('model.pt', 'rb') as f:
    model = torch.load(f)

with open('vocab.pickle', 'rb') as handle:
    vocab = pickle.load(handle)

inverted_vocab = invert_vocab(vocab)
vocab_size = len(vocab.keys())
hidden = model.initHidden()
sentence_beginning = args.prefix
words = ('<start> ' + sentence_beginning).split()
try:
    mini_batch_size = model.mini_batch_size
except AttributeError:
    mini_batch_size = 1

max_len = 50
i = 0
word = '<start>'
while word != '<end>' and i < max_len:
    input_tensor = input_tensor_from_list(words, vocab, vocab_size)
    #input_list = [input_tensor for j in range(mini_batch_size)]
    #input_batch = Variable(torch.cat(input_list).view(1, mini_batch_size, vocab_size))
    outputs, hidden = model(input_tensor, hidden, input_tensor.size()[0])
    word = decode_word(outputs[-1][0].view(1, outputs.size()[2]), inverted_vocab)
    words.append(word)
    i += 1

print ' '.join(words[1:-1])
