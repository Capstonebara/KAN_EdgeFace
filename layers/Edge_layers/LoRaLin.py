import torch.nn as nn
from layers.KAN_layers.KANLinear import KANLinear

class LoRaLin_KAN(nn.Module):
    def __init__(self, in_features, out_features, rank_ratio=0.5):
        super(LoRaLin_KAN, self).__init__()
        rank = max(2,int(min(in_features, out_features) * rank_ratio))
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.linear1 = KANLinear(in_features, rank)
        self.linear2 = KANLinear(rank, out_features)

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        return x


class LoRaLin(nn.Module):
    def __init__(self, in_features, out_features, rank_ratio=0.2, bias=True):
        super(LoRaLin, self).__init__()
        rank = max(2,int(min(in_features, out_features) * rank_ratio))
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        return x
