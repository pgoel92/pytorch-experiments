import torch
import torch.nn as nn
from torch.autograd import Variable

class myRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, mini_batch_size):
        super(myRNN, self).__init__()

        self.encoder = nn.RNN(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.outputs = nn.LogSoftmax(dim=2)

        self.hidden_size = hidden_size

    # inputs : sentence_len x mini_batch_size x input_size
    # hidden : 1 x hidden_size
    def forward(self, inputs, hidden, steps):
        interms = self.encoder(inputs)
        interms2 = self.decoder(interms[0])
        outputs = self.outputs(interms2)
   
        return outputs, Variable(torch.zeros(1, self.hidden_size))

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
