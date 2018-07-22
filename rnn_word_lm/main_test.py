from __future__ import unicode_literals
from torch.autograd import Variable
from io import open
import torch
import torch.nn as nn
import random 
import pickle
from model import RNNModel as mymodel
from pymodel import RNNModel as pymodel
import math
import timeit
import argparse
from evaluate import evaluate
from generate import generate
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--bsz', type=int, default=20, help='batch size')
parser.add_argument('--nhid', type=int, default=200, help='number of hidden units')
parser.add_argument('--vsize', type=int, default=2997, help='max vocab size')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--data', type=str, default='./data/fiftyk', help='location of data corpus')
parser.add_argument('--model', type=str, default='mymodel', help='location of data corpus')
args = parser.parse_args()
device = torch.device("cuda" if args.cuda else "cpu")
inverted_vocab = {}
criterion = nn.CrossEntropyLoss()
bptt = 35

def invert_vocab(vocab):
    inverted_vocab = {}
    for key, value in vocab.iteritems():
        inverted_vocab[value] = key 

    return inverted_vocab

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def replace_words_with_indices(words, vocab):
    return [vocab[word] for word in words]

def readData(filename, max_vocab_size):
    lines = open(filename, encoding='utf-8').readlines()
    lines = [line.strip() for line in lines]
    lines = [line.lower() for line in lines]
    lines = [add_start_end(line) for line in lines]
    words = ' '.join(lines).split()
    word_dict = {}
    for word in words:
        if word not in word_dict:
            word_dict[word] = 1
        else:
            word_dict[word] += 1

    words_sorted_alphabetically = sorted(word_dict.iteritems(), key = lambda x: x[0])
    words_sorted_by_count = sorted(word_dict.iteritems(), key = lambda x: x[1], reverse=True)

    vocab = {}
    i = 0
    for word, frequency in words_sorted_by_count[:min(max_vocab_size, len(words_sorted_by_count))]:
        vocab[word] = i
        i += 1

    words = replace_unknown(words, vocab)

    vocab['<unk>'] = i
    vocab_size = i + 1

    words = replace_words_with_indices(words, vocab)
    words = torch.LongTensor(words)

    return words, vocab

def is_unknown(word, vocab):
    return word if word in vocab else '<unk>'

def add_start_end(line):
    return '<start> ' + line + ' <end>'

def replace_unknown(words, vocab):
    return [is_unknown(word, vocab) for word in words]

def get_batch_pytorch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz 
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

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

def train(rnn, hidden, data, targets):
    rnn.train()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=args.lr)
    rnn.zero_grad()

    hidden = repackage_hidden(hidden)
    output, hidden = rnn.forward(data, hidden)
    loss = criterion(output.view(-1, output.size()[2]), targets)
    loss.backward()
    optimizer.step()

    return

def evaluate(rnn, vocab):
    rnn.eval()
    words, test_vocab = readData(args.data + '/test.txt', vocab)
    test_data = batchify(words, args.bsz)
    total_loss = 0 
    k = 0
    for batch, i in enumerate(range(0, test_data.size(0) - 1, bptt)):
        data, targets = get_batch_pytorch(test_data, i)
        hidden = rnn.initHidden(args.bsz, device)

        output, hidden = rnn.forward(data, hidden)
        loss = criterion(output.view(-1, output.size()[2]), targets).item()
        total_loss += loss
        k += 1

    loss = total_loss/k

    return loss, math.exp(loss)

def generate(rnn, vocab_size, inverted_vocab):
    hidden = rnn.initHidden(1, device)
    input = torch.randint(vocab_size, (1, 1), dtype=torch.long).to(device)
   
    words = []; 
    with torch.no_grad():  # no tracking history
        for i in range(20):
            output, hidden = rnn.forward(input, hidden)
            word_weights = output.squeeze().exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)
            word = inverted_vocab[word_idx.item()]
            words.append(word)
        
    print "Generated sample:"
    print ' '.join(words)

def main():
    words, vocab = readData(args.data + '/train.txt', args.vsize)
    inverted_vocab = invert_vocab(vocab)
    vocab_size = len(vocab.keys())
    print("Vocabulary size : " + str(vocab_size))
    train_data = batchify(words, args.bsz)
    print train_data.size()
    with open('vocab.pickle', 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if args.model == 'mymodel':
        rnn = mymodel('LSTM', vocab_size, args.nhid, args.nhid, 2, 0.2).to(device)
    else:
        rnn = pymodel('LSTM', vocab_size, args.nhid, args.nhid, 2, 0.2).to(device)
    learning_rate = args.lr

    running_loss = 0
    num_epochs = 40
    start_time = timeit.default_timer()
    training_loss = []
    dev_loss = []
    dev_perplexity = []
    test_loss = []
    prev_dev_perplexity = 9999999999;

    k = 0
    for e in range(num_epochs):
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch_pytorch(train_data, i)
            hidden = rnn.initHidden(args.bsz, device)
            train(rnn, hidden, data, targets)
            k += 1
        elapsed = timeit.default_timer() - start_time
        print('##################')
        print('Epoch %d :' % e)
        print('Time elapsed : %s' % (get_readable_time(int(elapsed))))
        loss, perp = evaluate(rnn, vocab)

        #dev_loss.append(loss)
        #dev_perplexity.append(perp)
        print('Validation loss : %.1f' % loss)
        print('Validation perplexity : %.1f' % perp)
        generate(rnn, vocab_size, inverted_vocab)
        with open('model.pt', 'wb') as f:
            torch.save(rnn, f)
        #if perp > prev_dev_perplexity:
        #    learning_rate /= 4
        #prev_dev_perplexity = perp
    #plotTrainingVsDevLoss(training_loss, dev_loss, 'training_vs_dev_loss.png')

if __name__ == "__main__":
    print "V3.1"
    main()
