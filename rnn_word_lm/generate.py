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
import random

parser = argparse.ArgumentParser()
parser.add_argument('--prefix')
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

def invert_vocab(vocab):
    inverted_vocab = {}
    for key, value in vocab.iteritems():
        inverted_vocab[value] = key

    return inverted_vocab

def encode_word(word, vocab):
    vocab_size = len(vocab.keys())
    word_tensor = torch.zeros(1, 1, vocab_size)
    if word not in vocab:
        word_tensor[0][0][vocab_size - 1] = 1
    else:
        word_tensor[0][0][vocab[word]] = 1

    return word_tensor

def decode_word(output, inverted_vocab):
    word_idx = torch.multinomial(output.exp(), 1)[0][0].item()
    return inverted_vocab[word_idx]
    #print output.exp()[0].topk(5)
    #topv, topi = output.data.topk(1)
    #topi = topi[0][0]
    #print topi.item()
    #word = inverted_vocab[topi.item()]
    return word

def input_tensor_from_list(ls, vocab, vocab_size):
    nwords = len(ls)
    input_tensor = torch.zeros(nwords, 1, vocab_size) 
    for i in range(len(ls)):
        inp = encode_word(ls[i], vocab)
        input_tensor[i] = inp

    return Variable(input_tensor)

def random_word(vocab):
    words = vocab.keys()
    word = words[random.randint(0, len(words))]
    return word, encode_word(word, vocab)

with open('model.pt', 'rb') as f:
    model = torch.load(f)

with open('vocab.pickle', 'rb') as handle:
    vocab = pickle.load(handle)

inverted_vocab = invert_vocab(vocab)
vocab_size = len(vocab.keys())
hidden = model.initHidden()
sentence_beginning = args.prefix
#words = ('<start> ' + sentence_beginning).split()
words = []

max_len = 50

for s in range(20):
    i = 0
    word, input_tensor = random_word(vocab)
    print word,
    while word != '<end>' and i < max_len:
        if args.cuda:
            input_tensor = input_tensor.cuda()
        outputs, hidden = model(input_tensor, hidden)
        word = decode_word(outputs[0], inverted_vocab)
        print word,
        input_tensor = encode_word(word, vocab)
        i += 1
    print
