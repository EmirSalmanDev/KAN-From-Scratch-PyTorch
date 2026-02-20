import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = 1.0 / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h)
        grid = grid.expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features * (grid_size + spline_order))
        )
        self.spline_scaler = nn.Parameter(torch.ones(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

    def b_splines(self, x):
        x = x.unsqueeze(-1)
        grid = self.grid

        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()

        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k + 1)]) /
                (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]
                +
                (grid[:, k + 1:] - x) /
                (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )

        return bases.contiguous()

    def forward(self, x):
        x = x.view(-1, self.in_features)

        base_out = F.linear(F.silu(x), self.base_weight)

        x_norm = torch.clamp(x, -1, 1)
        spline_basis = self.b_splines(x_norm)
        spline_out = F.linear(
            spline_basis.view(x.size(0), -1),
            self.spline_weight
        )

        out = base_out + self.spline_scaler * spline_out
        return out


class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(len(layers_hidden) - 1):
            self.layers.append(
                KANLayer(
                    layers_hidden[i],
                    layers_hidden[i + 1],
                    grid_size=grid_size
                )
            )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x