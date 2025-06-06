# models/exercise_lstm.py

import torch
import torch.nn as nn

class ExerciseLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=128, num_classes=2):
        super(ExerciseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.hidden_size = hidden_size

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Use output from last time step
        return out
