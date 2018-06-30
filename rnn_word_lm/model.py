import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    def __init__(self, model_type, vocab_size, emb_size, hidden_size, nlayers, dropout):
        super(RNNModel, self).__init__()

        print("My RNN model")
        self.encoder = nn.LSTM(vocab_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(0.2)

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        #self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform(-initrange, initrange)

    def indexToTensor(self, index):
        x = torch.zeros(self.vocab_size)
        x[index] = 1
        return x

    def tokensToTensors(self, input):
        l, b = input.size() 
        x = torch.FloatTensor(l, b, self.vocab_size)
        for i in range(l):
            for j in range(b):
                x[i][j] = self.indexToTensor(input[i][j])

        return x.to(input.device)
        
    # inputs : sentence_len x mini_batch_size x input_size
    # hidden : 1 x hidden_size
    def forward(self, input, hidden=None):
        input_tensors = self.tokensToTensors(input)
        output, hidden = self.encoder(input_tensors, hidden)
        output = self.drop(output)
        outputs = self.decoder(output)
   
        return outputs, hidden

    def initHidden(self, bsz, device):
        return (torch.zeros(1, bsz, self.hidden_size).to(device),
                torch.zeros(1, bsz, self.hidden_size).to(device))
