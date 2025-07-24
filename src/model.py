
import torch


class ModelOne(torch.nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, emb_dim)
        self.lstm1 = torch.nn.LSTM(emb_dim, 16, num_layers=1,batch_first=True)
        self.fc1 = torch.nn.Linear(16, 5)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        z = self.emb(x)
        z = torch.nn.utils.rnn.pack_padded_sequence(z,lengths, batch_first=True, enforce_sorted=False)
        _, (h_,c_) = self.lstm1(z)
        z = self.dropout(h_[0])
        z = self.fc1(z)
        return z

