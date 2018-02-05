from __future__ import unicode_literals
from torch.autograd import Variable
from io import open
import torch
import torch.nn as nn
import random 
import pickle
import model
import math
import timeit

MAX_VOCAB_SIZE = 20000
mini_batch_size = 250 

vocab = {}
def encode_word(word):
    vocab_size = len(vocab.keys())
    word_tensor = torch.zeros(1, vocab_size)
    if word not in vocab:
        word_tensor[0][vocab_size - 1] = 1
    else:
        word_tensor[0][vocab[word]] = 1

    return word_tensor

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    return Variable(h.data)

def train(rnn, hidden, criterion, learning_rate, input_batch, target_batch):
    rnn.zero_grad()

    hidden = repackage_hidden(hidden)
    outputs, hidden = rnn.forward(input_batch, hidden, input_batch.size()[0])
    loss = criterion(outputs.view(-1, outputs.size()[2]), target_batch.view(-1))
    loss.backward()

    torch.nn.utils.clip_grad_norm(rnn.parameters(), 0.25)
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return outputs, loss.data[0], hidden

def readData():
    lines = open('input.txt', encoding='utf-8').readlines()
    lines = [line.strip() for line in lines]
    lines = [line.lower() for line in lines]
    words = ' '.join(lines).split()
    word_dict = {}
    for word in words:
        if word not in word_dict:
            word_dict[word] = 1
        else:
            word_dict[word] += 1

    words_sorted_alphabetically = sorted(word_dict.iteritems(), key = lambda x: x[0])
    words_sorted_by_count = sorted(word_dict.iteritems(), key = lambda x: x[1], reverse=True)

    i = 0
    for word, frequency in words_sorted_by_count[:min(MAX_VOCAB_SIZE, len(words_sorted_by_count))]:
        vocab[word] = i
        i += 1

    lines = [replace_unknown(line) for line in lines]

    vocab['<start>'] = i 
    vocab['<end>'] = i + 1
    vocab['<unk>'] = i + 2
    vocab_size = i + 3
    return lines, vocab_size

def is_unknown(word):
    return word if word in vocab else '<unk>'

def replace_unknown(line):
    words = line.split()
    words = [is_unknown(word) for word in words] 
    return ' '.join(words)

def listToTensor(ls):
    return torch.stack(ls)

def lineToTensor(line):
    words = line.split()
    word_tensors = map(encode_word, words)
    input_tensor = listToTensor([encode_word('<start>')] + word_tensors[:len(word_tensors)])
    target_tensor = torch.LongTensor([vocab[word] for word in words] + [vocab['<end>']])

    return input_tensor, target_tensor

def get_batch(lines, batch_number, vocab_size):
    batch_lines = lines[batch_number * 5:batch_number * 5 + mini_batch_size]
    input_batch = torch.zeros(mini_batch_size, 21, vocab_size)
    target_batch = torch.zeros(mini_batch_size, 21).type(torch.LongTensor)
    for i in range(mini_batch_size):
        input, target = lineToTensor(batch_lines[i])
        input_batch[i] = input.squeeze()
        target_batch[i] = target

    return Variable(input_batch.view(21, mini_batch_size, vocab_size)), Variable(target_batch.view(21, mini_batch_size))

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(lines):
    line = randomChoice(lines)
    input_line_tensor, target_line_tensor = lineToTensor(line)

    return input_line_tensor, target_line_tensor

def get_readable_time(secs):
    readable_time = ""
    if secs >= 3600*24:
        days = secs/(3600*24)
        readable_time += str(days) + "days"
        secs -= 3600*24*days
    if secs >= 3600:
        hrs = secs/3600
        readable_time += " " + str(hrs) + "hrs"
        secs -= 3600*hrs
    if secs >= 60:
        mins = secs/60
        readable_time += " " + str(mins) + "mins"
        secs -= 60*mins
    if secs > 0:
        readable_time += " " + str(secs) + "s"

    return readable_time.strip()

def main():
    lines, vocab_size = readData()
    print("Vocabulary size : " + str(vocab_size))
    with open('vocab.pickle', 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rnn = model.RNN(vocab_size, 128, vocab_size, mini_batch_size)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.75

    running_loss = 0
    num_epochs = 10000
    num_iterations = len(lines) / mini_batch_size
    print_every = num_iterations/10 - 1
    start_time = timeit.default_timer()
    hidden = rnn.initHidden()
    for e in range(num_epochs):
        for i in range(num_iterations):
            input_batch, target_batch = get_batch(lines, i, vocab_size)
            outputs, loss, hidden = train(rnn, hidden, criterion, learning_rate, input_batch, target_batch)

            # print(str(loss) + " # " + lines[i%len(lines)])
            running_loss += loss

        elapsed = timeit.default_timer() - start_time
        print('Epoch %d : %.4f' % (e, running_loss/num_iterations))
        #print('Time elapsed : %s, Projected epoch training time : %s' % (get_readable_time(int(elapsed)), get_readable_time(int((elapsed/(i + e*num_iterations))*num_iterations))))
        print('Time elapsed : %s' % (get_readable_time(int(elapsed))))
        running_loss = 0

        with open('model.pt', 'wb') as f:
            torch.save(rnn, f)


main()
