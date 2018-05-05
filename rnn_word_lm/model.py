import torch
import torch.nn as nn
from torch.autograd import Variable

class myRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, mini_batch_size):
        super(myRNN, self).__init__()

        self.encoder = nn.LSTM(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

        self.hidden_size = hidden_size

        #self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform(-initrange, initrange)

    # inputs : sentence_len x mini_batch_size x input_size
    # hidden : 1 x hidden_size
    def forward(self, inputs, hidden):
        interms = self.encoder(inputs, hidden)
        outputs = self.decoder(interms[0])
   
        return outputs, hidden

    def initHidden(self, bsz):
        return (Variable(torch.zeros(1, bsz, self.hidden_size)),
                Variable(torch.zeros(1, bsz, self.hidden_size)))
