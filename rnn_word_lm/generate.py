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

    return Variable(word_tensor)

def decode_word(output, inverted_vocab):
    topv, topi = output.data.topk(1)
    topi = topi[0][0]
    word = inverted_vocab[topi]
    return word

with open('model.pt', 'rb') as f:
    model = torch.load(f)

with open('vocab.pickle', 'rb') as handle:
    vocab = pickle.load(handle)

inverted_vocab = invert_vocab(vocab)
vocab_size = len(vocab.keys())
hidden = model.initHidden()
sentence_beginning = ''
words = ('<start> ' + sentence_beginning).split()
for word in words:
    input_tensor = encode_word(word, vocab)
    outputs, hidden = model(input_tensor, hidden, input_tensor.size()[0])
    word = decode_word(outputs[0], inverted_vocab)

decoded_sentence = sentence_beginning.split()
max_len = 100
i = 0
while word != '<end>' and i < max_len:
    decoded_sentence.append(word)
    input_tensor = encode_word(word, vocab)
    outputs, hidden = model(input_tensor, hidden, input_tensor.size()[0])
    word = decode_word(outputs[0], inverted_vocab)
    i += 1

print ' '.join(decoded_sentence)
