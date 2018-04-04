import torch
import torch.nn as nn
from torch.autograd import Variable

class myRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, mini_batch_size):
        super(myRNN, self).__init__()

        self.encoder = nn.RNN(input_size, hidden_size)
        self.decoder = nn.RNN(hidden_size, output_size)

        self.hidden_size = hidden_size

    # inputs : sentence_len x mini_batch_size x input_size
    # hidden : 1 x hidden_size
    def forward(self, inputs, hidden, steps):
        interms = self.encoder(inputs)
        outputs = self.decoder(interms[0])
   
        return outputs[0], Variable(torch.zeros(1, self.hidden_size))

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
