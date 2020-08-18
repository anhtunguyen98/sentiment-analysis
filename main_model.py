from torch.nn.functional import softmax
import torch
class Sentiment_Analysic(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_labels):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.bilstm = torch.nn.LSTM(bidirectional=True, input_size=embedding_dim, hidden_size=512, num_layers=2,
                                    batch_first=True)
        # self.dropout=torch.nn.Dropout(0.5)
        self.linear1 = torch.nn.Linear(512, 128)
        self.linear2 = torch.nn.Linear(128, num_labels)

    def forward(self, input):
        output = self.embedding(input)
        output, (hidden, cell) = self.bilstm(output)

        output = self.linear1(hidden[-1])
        output = self.linear2(output)
        # output=self.dropout(output)
        return softmax(output, dim=-1)


import torch.nn as nn
from torch.nn import functional as F


class Cnn_Sentiment_Analysis(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, window_sizes=(3, 4, 5)):
        super(Cnn_Sentiment_Analysis, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, 128, [window_size, embedding_dim], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])

        self.fc = nn.Linear(128 * len(window_sizes), 2)

    def forward(self, x):
        x = self.embedding(x)  # [B, T, E]

        # Apply a convolution + max pool layer for each window size
        x = torch.unsqueeze(x, 1)  # [B, C, T, E] Add a channel dim.
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))  # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)  # [B, F, window]

        # FC
        x = x.view(x.size(0), -1)  # [B, F * window]
        logits = self.fc(x)  # [B, class]

        # Prediction
        probs = F.softmax(logits, dim=-1)  # [B, class]

        return probs
