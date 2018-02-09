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

    return input_tensor, target_tensor

def get_batch(lines, batch_number, mini_batch_size, vocab_size, vocab):
    batch_lines = lines[batch_number * 5:batch_number * 5 + mini_batch_size]
    input_batch = torch.zeros(mini_batch_size, 21, vocab_size)
    target_batch = torch.zeros(mini_batch_size, 21).type(torch.LongTensor)
    for i in range(mini_batch_size):
        input, target = lineToTensor(batch_lines[i], vocab)
        input_batch[i] = input.squeeze()
        target_batch[i] = target

    return Variable(input_batch.view(21, mini_batch_size, vocab_size)), Variable(target_batch.view(21, mini_batch_size))

def evaluate():
    with open('vocab.pickle', 'rb') as handle:
        vocab = pickle.load(handle)

    with open('model.pt', 'rb') as f:
        model = torch.load(f)

    vocab_size = len(vocab.keys())
    criterion = nn.CrossEntropyLoss()
    test_lines = readTestData(vocab)
    total_loss = 0 
    number_of_batches = len(test_lines)/model.mini_batch_size
    for j in range(number_of_batches):
        input_batch, target_batch = get_batch(test_lines, j, model.mini_batch_size, vocab_size, vocab)

        hidden = model.initHidden()

        outputs, hidden = model(input_batch, hidden, input_batch.size()[0])
        loss = criterion(outputs.view(-1, outputs.size()[2]), target_batch.view(-1)).data[0]
        total_loss += loss

    loss = total_loss/number_of_batches
    print('Perplexity : %.1f' % math.exp(loss))

evaluate()
