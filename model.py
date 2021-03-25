from __future__ import print_function
import torch.nn as nn


class ANRL(nn.Module):
    def __init__(self, struct):
        super(ANRL, self).__init__()
        self.encode = Encoder(struct)
        self.decode = Decoder(struct)

    def ae_process(self, x):
        return self.decode(self.encode(x))

    def sg_process(self, x):
        return self.encode(x)


class Encoder(nn.Module):
    def __init__(self, struct):
        super(Encoder, self).__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(struct[i], struct[i + 1])
                                            for i in range(len(struct) - 1)])
        self.tanh = nn.Tanh()

    def forward(self, x):
        for linear in self.linear_layers:
            x = self.tanh(linear(x))
        return x


class Decoder(nn.Module):
    def __init__(self, struct):
        super(Decoder, self).__init__()
        struct.reverse()
        self.linear_layers = nn.ModuleList([nn.Linear(struct[i], struct[i + 1])
                                            for i in range(len(struct) - 1)])
        self.tanh = nn.Tanh()

    def forward(self, x):
        for linear in self.linear_layers:
            x = self.tanh(linear(x))
        return x
