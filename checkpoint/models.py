import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.autograd import Variable
from tensorflow.keras.preprocessing.sequence import pad_sequences


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, max_sequence_length, bidirectional, sum_output):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.max_sequence_length = max_sequence_length
        self.bidirectional = bidirectional
        self.sum_output = sum_output

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        if bidirectional:
            self.lstm = self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        x = pad_sequences(x, maxlen=self.max_sequence_length, padding='post', truncating='post')
        x = torch.Tensor(x).to(device)

        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        if self.bidirectional:
            h0 = Variable(torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size)).to(device)
            c0 = Variable(torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size)).to(device)
        
        out, (h, c) = self.lstm(x, (h0, c0))

        if self.sum_output:
            out = torch.sum(out, axis=1) # Sum of all lstm cell output to feed into linear layer
        else:
            out = out[:, -1, :] # get the output at final time step to feed into linear layer

        out = self.fc(out)
        return out


class LSTMEmbeddedModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, embedding_dim, max_sequence_length, bidirectional):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.max_sequence_length = max_sequence_length
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        if bidirectional:
            self.lstm = self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        x = pad_sequences(x, maxlen=self.max_sequence_length, padding='post', truncating='post')
        x = torch.LongTensor(x).to(device)
        x = self.embedding(x)

        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        if self.bidirectional:
            h0 = Variable(torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size)).to(device)
            c0 = Variable(torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size)).to(device)
        
        out, (h, c) = self.lstm(x, (h0, c0))

        out = torch.sum(out, axis=1) # get the output at final time step to feed into linear layer
        out = self.fc(out)

        return out


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                embedding, vocab_size, embedding_dim):
        super(SimpleRNN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = embedding
        if self.embedding:
            assert self.input_size == 1, 'Input must be a list of indices'
            assert isinstance(vocab_size, int), 'vocab_size must be int'
            assert isinstance(embedding_dim, int), 'embedding_dim must be int'
            self.embedding_layer = nn.Embedding(vocab_size, embedding_dim,
                                                    max_norm=1, device=device)
            self.input_size = embedding_dim
            self.input_to_hidden = nn.Linear(self.input_size,
                                                self.hidden_size,
                                                device=self.device)
            self.hidden_to_hidden = nn.Linear(self.hidden_size,
                                                self.hidden_size,
                                                device=self.device)
            self.hidden_to_output = nn.Linear(self.hidden_size,
                                                self.output_size,
                                                device=self.device)
            self.tanh = nn.Tanh()
            self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor):
        if self.embedding:
            input_tensor = self.embedding_layer(input_tensor.long()).squeeze()
        hidden = torch.zeros((1, self.hidden_size), device=self.device)
        for word in input_tensor:
            hidden = self.input_to_hidden(word) + self.hidden_to_hidden(hidden)
            hidden = self.tanh(hidden)
            output = self.hidden_to_output(hidden)
        # Only the output of the last RNN cell is taken into account
        output = self.softmax(output)
        # The log-softmax function combined with negative log likelihood loss
        # gives the same effect as cross entropy loss taken straight from the output
        return output

    def predict(self, input_tensor):
        with torch.no_grad():
            prediction = torch.argmax(self.forward(input_tensor))
        return prediction


class DeepRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                embedding, vocab_size, embedding_dim):
        super(DeepRNN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = embedding
        self.embedding_dim = embedding_dim
        if self.embedding:
            assert self.input_size == 1, 'Input must be a list of indices'
            assert isinstance(vocab_size, int), 'vocab_size must be int'
            assert isinstance(embedding_dim, int), 'embedding_dim must be int'
            self.embedding_layer = nn.Embedding(vocab_size, embedding_dim,
                                                max_norm=1, device=device)
            self.input_size = embedding_dim
        self.input_to_hidden = nn.Linear(self.input_size,
                                            self.hidden_size,
                                            device=self.device)
        self.h2h_same_layer_1 = nn.Linear(self.hidden_size,
                                            self.hidden_size,
                                            device=self.device)
        self.h2h_between_layers_12 = nn.Linear(self.hidden_size,
                                                self.hidden_size,
                                                device=self.device)
        self.h2h_same_layer_2 = nn.Linear(self.hidden_size,
                                            self.hidden_size,
                                            device=self.device)
        self.h2h_between_layers_23 = nn.Linear(self.hidden_size,
                                                self.hidden_size,
                                                device=self.device)
        self.h2h_same_layer_3 = nn.Linear(self.hidden_size,
                                            self.hidden_size,
                                            device=self.device)
        self.hidden_to_output = nn.Linear(self.hidden_size,
                                            self.output_size,
                                            device=self.device)
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor):
        if self.embedding:
            input_tensor = self.embedding_layer(input_tensor.long()).squeeze()
        # Each layer has a hidden state
        hidden_1 = torch.zeros((1, self.hidden_size), device=self.device)
        hidden_2 = torch.zeros((1, self.hidden_size), device=self.device)
        hidden_3 = torch.zeros((1, self.hidden_size), device=self.device)
        for word in input_tensor:
            hidden_1 = self.input_to_hidden(word) + self.h2h_same_layer_1(hidden_1)
            hidden_1 = self.tanh(hidden_1)
            hidden_2 = self.h2h_between_layers_12(hidden_1) + self.h2h_same_layer_2(hidden_2)
            hidden_2 = self.tanh(hidden_2)
            hidden_3 = self.h2h_between_layers_23(hidden_2) + self.h2h_same_layer_3(hidden_3)
            hidden_3 = self.tanh(hidden_3)
            output = self.hidden_to_output(hidden_3)
        # Only the output of the last RNN cell is taken into account
        output = self.softmax(output)
        # The log-softmax function combined with negative log likelihood loss
        # gives the same effect as cross entropy loss taken straight from the output
        return output

    def predict(self, input_tensor):
        with torch.no_grad():
            prediction = torch.argmax(self.forward(input_tensor))
        return prediction


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:

        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float, sequence_size: int, class_num: int):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout, sequence_size)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, class_num)
        # self.avgpool = nn.AvgPool1d(sequence_size)
        self.weighted_pool = nn.Linear(sequence_size,1)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, pretraining = False) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        if pretraining:
            output = self.decoder(output)
        else:
            output = self.linear(output)
            # output = self.avgpool(output.permute(2,1,0)).permute(2,1,0)
            output = self.weighted_pool(output.permute(2,1,0)).permute(2,1,0)
        output = F.log_softmax(output,dim =-1)
        return output.squeeze()