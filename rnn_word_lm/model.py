import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.lin_xh = nn.Linear(input_size, hidden_size)
        self.lin_hh = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.lin_ho = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def step(self, input, hidden):
        self.hidden = self.sigmoid(self.lin_xh(input) + self.lin_hh(hidden))
        self.output = self.lin_ho(self.hidden)
        self.output = self.softmax(self.output)

    def forward(self, inputs, hidden, steps):
        outputs = Variable(torch.zeros(steps, 1, self.input_size))
        self.hidden = hidden
        for i in range(steps):
            self.step(inputs[i], self.hidden)
            outputs[i] = self.output

        return outputs, self.hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
