import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layers_hidden):
        super().__init__()

        layers = []
        for i in range(len(layers_hidden) - 1):
            layers.append(nn.Linear(layers_hidden[i], layers_hidden[i + 1]))

            if i < len(layers_hidden) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(layers_hidden[i + 1]))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)