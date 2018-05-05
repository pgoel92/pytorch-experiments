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
import argparse
from evaluate import evaluate
from generate import generate
import matplotlib
import matplotlib.pyplot as plt

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
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def readData(filename, max_vocab_size):
    lines = open(filename, encoding='utf-8').readlines()
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
    for word, frequency in words_sorted_by_count[:min(max_vocab_size, len(words_sorted_by_count))]:
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

def get_batch(lines, batch_number, num_tokens, vocab_size, batch_size):
    batch_lines = lines[batch_number * batch_size:(batch_number + 1) * batch_size]
    # The number of steps is num_tokens + 1 because of <start> and <end> being appended to each input and target sequence respectively
    input_batch = torch.zeros(batch_size, num_tokens + 1, vocab_size)
    target_batch = torch.zeros(batch_size, num_tokens + 1).type(torch.LongTensor)
    for i in range(batch_size):
        input, target = lineToTensor(batch_lines[i])
        input_batch[i] = input.squeeze()
        target_batch[i] = target

    return Variable(input_batch.view(num_tokens + 1, batch_size, vocab_size)), Variable(target_batch.view(num_tokens + 1, batch_size))

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

def plotTrainingVsDevLoss(training_loss, dev_loss, filename):
    """ Make a plot of regularization vs accuracy """
    plt.plot(xrange(len(training_loss)), training_loss)
    plt.plot(xrange(len(dev_loss)), dev_loss)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.legend(['train', 'dev'], loc='upper left')
    plt.savefig(filename)

def train(rnn, hidden, criterion, learning_rate, input_batch, target_batch):
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    rnn.zero_grad()

    hidden = repackage_hidden(hidden)
    outputs, hidden = rnn.forward(input_batch, hidden)
    loss = criterion(outputs.view(-1, outputs.size()[2]), target_batch.view(-1))
    loss.backward()
    optimizer.step()

    #for p in rnn.parameters():
    #    p.data.add_(-learning_rate, p.grad.data)

    return outputs, loss.item(), hidden

def main(args):
    lines, vocab_size = readData(args.data + '/train.txt', args.vsize)
    print("Vocabulary size : " + str(vocab_size))
    with open('vocab.pickle', 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rnn = model.myRNN(vocab_size, args.nhid, vocab_size, args.bsz)
    if args.cuda:
        rnn.cuda()
    criterion = nn.CrossEntropyLoss()
    learning_rate = args.lr

    running_loss = 0
    num_epochs = 300
    num_iterations = len(lines) / args.bsz
    print_every = num_iterations/10 - 1
    start_time = timeit.default_timer()
    #hidden = rnn.initHidden(args.bsz)
    num_tokens = len(lines[0].split())
    training_loss = []
    dev_loss = []
    dev_perplexity = []
    test_loss = []
    prev_dev_perplexity = 9999999999;
    for e in range(num_epochs):
        for i in range(num_iterations):
            hidden = rnn.initHidden(args.bsz)
            input_batch, target_batch = get_batch(lines, i, num_tokens, vocab_size, args.bsz)
            if args.cuda:
                input_batch = input_batch.cuda()
                target_batch = target_batch.cuda()
            outputs, loss, hidden = train(rnn, hidden, criterion, learning_rate, input_batch, target_batch)

            # print(str(loss) + " # " + lines[i%len(lines)])
            running_loss += loss

        elapsed = timeit.default_timer() - start_time
        print('##################')
        print('Epoch %d :' % e)
        #print('Time elapsed : %s, Projected epoch training time : %s' % (get_readable_time(int(elapsed)), get_readable_time(int((elapsed/(i + e*num_iterations))*num_iterations))))
        print('Time elapsed : %s' % (get_readable_time(int(elapsed))))
        print('Training loss : %.4f' % (running_loss/num_iterations))
        training_loss.append(running_loss/num_iterations)
        running_loss = 0

        if args.cuda:
            loss, perp = evaluate(args.data + '/validation.txt', rnn, vocab, cuda=True)
        else:
            loss, perp = evaluate(args.data + '/validation.txt', rnn, vocab)
        dev_loss.append(loss)
        dev_perplexity.append(perp)
        print('Validation loss : %.1f' % loss)
        print('Validation perplexity : %.1f' % perp)
        samples = generate(rnn, vocab, args.cuda)
        print('Samples : ')
        for sample in samples:
            print(sample)
        with open('model.pt', 'wb') as f:
            torch.save(rnn, f)
        if perp > prev_dev_perplexity:
            learning_rate /= 4
        prev_dev_perplexity = perp
    plotTrainingVsDevLoss(training_loss, dev_loss, 'training_vs_dev_loss.png')

if __name__ == "__main__":
    print "V2.9"
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--bsz', type=int, default=20, help='batch size')
    parser.add_argument('--nhid', type=int, default=200, help='number of hidden units')
    parser.add_argument('--vsize', type=int, default=2997, help='max vocab size')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--data', type=str, default='./data/fiftyk', help='location of data corpus')
    args = parser.parse_args()
    main(args)
