import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, mini_batch_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.mini_batch_size = mini_batch_size

        self.lin_xh = nn.Linear(input_size, hidden_size)
        self.lin_hh = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.lin_ho = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    # input : mini_batch_size x input_size
    # input : mini_batch_size x hidden_size
    def step(self, input, hidden):
        xh = self.lin_xh(input) # mini_batch_size x hidden_size
        hh = self.lin_hh(hidden) # mini_batch_size x hidden_size
        hidden = self.sigmoid(xh + hh) # mini_batch_size x hidden_size
        output = self.lin_ho(hidden) # mini_batch_size x output_size
        output = self.softmax(output)

        return output, hidden

    # inputs : sentence_len x mini_batch_size x input_size
    # hidden : 1 x hidden_size
    def forward(self, inputs, hidden, steps):
        outputs = Variable(torch.zeros(steps, self.mini_batch_size, self.output_size))
        hidden_list = [hidden for i in range(self.mini_batch_size)]
        hidden_batch = torch.cat(hidden_list)
        for i in range(steps):
            output, hidden_batch = self.step(inputs[i], hidden_batch)
            outputs[i] = output
   
        self.hidden = torch.mean(hidden_batch, 0).view(1, self.hidden_size)
        return outputs, self.hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
