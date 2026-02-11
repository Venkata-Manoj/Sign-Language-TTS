import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=42, hidden_size=64, num_layers=2, num_classes=9):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out