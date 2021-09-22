import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


class Encoder_all_seq(nn.Module):
    """
        The class for LSTM encoder with the final output as the combination of the output for each date
    """

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.):
        """ Initilize LSTM encoder
        Args:
        input_dim: int -- the size/dimensionality of the vectors that will be input to the encoder
        hidden_dim: int -- the dimensionality of the hidden and cell states
        num_layers: int -- the number of layers in the LSTM
        dropout: float -- the amount of dropout to use
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

    def forward(self, src):
        """
        Forward function, return output from all steps in the input sequence
        """
        #  src = [batch size, time, dim]
        outputs, (hidden, cell) = self.lstm(src)
        return outputs


class EncoderFNN_AllSeq(nn.Module):
    """The class for an encoder (LSTM)-decoder(FNN) model where the input of the decoder is the output of all steps in input sequence
    """
    def __init__(self, input_dim, output_dim, hidden_dim=10, num_layers=2, seq_len=4, linear_dim=100, learning_rate=0.01, dropout=0.1, threshold=0.1, num_epochs=100):
        """ Initilize an encoder (LSTM)-decoder(FNN) model where the input of the decoder is the output of all steps in input sequence
        Args:
        input_dim: int -- the size/dimensionality of the vectors that will be input for encoder
        output_dim: int -- the size/dimensionality of the vectors that will be input for decoder
        hidden_dim: int -- the dimensionality of the hidden and cell states
        num_layers: int -- the number of layers in the LSTM
        seq_len: int -- the length of input sequence
        linear_dim: int -- the dimensionality of the decoder FNN
        learning_rate: float -- learning learning_rate
        threshold: float -- the early stopping point (training loss < threshold) for training process
        num_epochs: int -- the maximum number of training epochs
        """
        super().__init__()

        self.input_dim = input_dim  # batch_size seq_length n_features
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.threshold = threshold
        self.seq_len = seq_len
        self.dropout = dropout
        self.linear_dim = linear_dim

        self.encoder = Encoder_all_seq(self.input_dim, self.hidden_dim, self.num_layers, self.dropout)

        self.out1 = nn.Linear(self.hidden_dim * self.seq_len, self.linear_dim)
        self.out2 = nn.Linear(self.linear_dim, self.output_dim)

        # assert self.encoder.hidden_dim == self.decoder.hidden_dim, \
        #     "Hidden dimensions of encoder and decoder must be equal!"
        # assert self.encoder.num_layers == self.decoder.num_layers, \
        #     "Encoder and decoder must have equal number of layers!"

    def forward(self, src, device):
        """Forward function
        """
        # src = [batch size,sent len, src] e.g.[5,10,1]
        # trg = [batch size,sent len, trg] e.g.[5,10,1]

        src = torch.as_tensor(src[:, -self.seq_len:, :]).float()
        src = src.to(device)
        batch_size = src.shape[0]  # batch_size
        seq_len = src.shape[1]

        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size,self.decoder_len,self.output_dim).to(device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_output = self.encoder(src)

        # encoder_output = encoder_output.permute(1,0,2)

        encoder_output = encoder_output.reshape(encoder_output.shape[0], encoder_output.shape[1] * encoder_output.shape[2])

        linear_output = self.out1(encoder_output)
        linear_output = F.relu(linear_output)
        output = self.out2(linear_output)

        return output

    def fit(self, train_loader, device):
        """ Fit function for model training
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        criterion = torch.nn.MSELoss(reduction='mean')  # sum of the error for all element in the batch
        schedular = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

        for epoch in range(self.num_epochs):

            self.train()
            train_epoch_loss = 0

            for i, (src, trg) in enumerate(train_loader):
                src = torch.as_tensor(src).float()
                src = src.to(device)
                if len(src.size()) < 3:
                    src = src.view(1, -1)
                trg = torch.as_tensor(trg).float()
                trg = trg.to(device)

                train_output = self.forward(src, device)  # 1x197

                # loss = criterion(output[:,1:,:], trg[:,1:,:])
                loss = criterion(train_output, trg)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()
            schedular.step()

            print('Epoch: {}/{} Train Loss: {:.4f}'.format(epoch, self.num_epochs, train_epoch_loss / (i + 1)))

            if train_epoch_loss / (i + 1) < self.threshold:
                break

    def fit_cv(self, train_loader, val_src, val_trg, device):
        """ Fit function for hyper-parameter tuning
        """
        val_src = torch.as_tensor(val_src).float()
        val_trg = torch.as_tensor(val_trg).float()
        val_src = val_src.to(device)
        val_trg = val_trg.to(device)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss(reduction='mean')  # sum of the error for all element in the batch
        history = np.zeros((self.num_epochs, 2))
        schedular = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

        for epoch in range(self.num_epochs):
            self.train()
            train_epoch_loss = 0
            for i, (src, trg) in enumerate(train_loader):
                src = torch.as_tensor(src).float()
                train_output = self.forward(src, device)
                trg = torch.as_tensor(trg).float()
                trg = trg.to(device)
                loss = criterion(train_output, trg)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()
            # on validation set
            schedular.step()
            self.eval()
            val_output = self.forward(val_src, device)
            loss = criterion(val_output, val_trg)
            val_epoch_loss = loss.item()
            history[epoch] = [train_epoch_loss / (i + 1), val_epoch_loss]
            print('Epoch: {}/{} Train Loss: {:.4f} Validation Loss:{:.4f}'.format(epoch, self.num_epochs, train_epoch_loss / (i + 1), val_epoch_loss))
            if train_epoch_loss / (i + 1) < self.threshold:
                break
        return history[:epoch]

    # make prediction
    def predict(self, src, device):
        """ Predict function for a trained model to predict
        """
        self.eval()
        src = torch.as_tensor(src).float()
        src = src.to(device)

        return self.forward(src, device).detach().cpu().numpy()
