from io import open
import torch
import torch.nn as nn
import pickle
import math
from torch.autograd import Variable

def is_unknown(word, vocab):
    return word if word in vocab else '<unk>'

def replace_unknown(line, vocab):
    words = line.split()
    words = [is_unknown(word, vocab) for word in words]
    return ' '.join(words)

def readTestData(vocab):
    lines = open('test.txt', encoding='utf-8').readlines()
    lines = [line.strip() for line in lines]
    lines = [line.lower() for line in lines]
    lines = [replace_unknown(line, vocab) for line in lines]
    return lines

def encode_word(word, vocab):
    vocab_size = len(vocab.keys())
    word_tensor = torch.zeros(1, vocab_size)
    if word not in vocab:
        word_tensor[0][vocab_size - 1] = 1
    else:
        word_tensor[0][vocab[word]] = 1

    return word_tensor

def listToTensor(ls):
    return torch.stack(ls)

def lineToTensor(line, vocab):
    words = line.split()
    vocabs = [vocab for i in range(len(words))]
    word_tensors = map(encode_word, words, vocabs)
    input_tensor = listToTensor([encode_word('<start>', vocab)] + word_tensors[:len(word_tensors)])
    target_tensor = torch.LongTensor([vocab[word] for word in words] + [vocab['<end>']])

    return Variable(input_tensor), Variable(target_tensor)

def evaluate():
    with open('vocab.pickle', 'rb') as handle:
        vocab = pickle.load(handle)

    with open('model.pt', 'rb') as f:
        model = torch.load(f)

    vocab_size = len(vocab.keys())
    criterion = nn.CrossEntropyLoss()
    test_lines = readTestData(vocab)
    total_loss = 0 
    for line in test_lines:
        input_tensor, target_tensor = lineToTensor(line, vocab) 

        hidden = model.initHidden()

        outputs, hidden = model(input_tensor, hidden, input_tensor.size()[0])
        loss = criterion(outputs.squeeze(), target_tensor).data[0]
        total_loss += loss

    loss = total_loss/len(test_lines)
    print('Perplexity : %.1f' % math.exp(loss))

evaluate()
