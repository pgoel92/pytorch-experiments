from io import open
import torch
import torch.nn as nn
import pickle
import math
from torch.autograd import Variable
import argparse

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
args = parser.parse_args()

def is_unknown(word, vocab):
    return word if word in vocab else '<unk>'

def replace_unknown(line, vocab):
    words = line.split()
    words = [is_unknown(word, vocab) for word in words]
    return ' '.join(words)

def readTestData(filename, vocab):
    lines = open(filename, encoding='utf-8').readlines()
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

def get_batch(lines, batch_number, mini_batch_size, num_tokens, vocab_size, vocab):
    batch_lines = lines[batch_number * mini_batch_size:(batch_number + 1) * mini_batch_size]
    input_batch = torch.zeros(mini_batch_size, num_tokens + 1, vocab_size)
    target_batch = torch.zeros(mini_batch_size, num_tokens + 1).type(torch.LongTensor)
    for i in range(mini_batch_size):
        input, target = lineToTensor(batch_lines[i], vocab)
        input_batch[i] = input.squeeze()
        target_batch[i] = target

    return Variable(input_batch.view(num_tokens + 1, mini_batch_size, vocab_size)), Variable(target_batch.view(num_tokens + 1, mini_batch_size))

def evaluate(filename, model=None, vocab=None, cuda=False):
    if not vocab:
        with open('vocab.pickle', 'rb') as handle:
            vocab = pickle.load(handle)

    if not model:
        with open('model.pt', 'rb') as f:
            model = torch.load(f)

    vocab_size = len(vocab.keys())
    criterion = nn.CrossEntropyLoss()
    test_lines = readTestData(filename, vocab)
    num_tokens = len(test_lines[0].split())
    total_loss = 0 
    for j in range(len(test_lines)):
        input_batch, target_batch = get_batch(test_lines, j, 1, num_tokens, vocab_size, vocab)
        if cuda:
            input_batch = input_batch.cuda()
            target_batch = target_batch.cuda()
            model.cuda()

        hidden = model.initHidden()

        outputs, hidden = model(input_batch, hidden)
        loss = criterion(outputs.view(-1, outputs.size()[2]), target_batch.view(-1)).item()
        total_loss += loss

    loss = total_loss/len(test_lines)

    return loss, math.exp(loss)

if __name__ == "__main__":
    if args.cuda:
        loss, perp = evaluate('test.txt', cuda=True)
    else:
        loss, perp = evaluate('test.txt')
    print('Loss : %.1f' % loss)
    print('Perplexity : %.1f' % perp)
