import torch
import numpy as np

class ElmoCombiner(torch.nn.Module):
    """
    Simple linear combination of the embeddings
    used by elmo.
    """

    def __init__(self, input_layer=3, emb_dimension=1024, freeze=False):
        super(ElmoCombiner, self).__init__()
        self.input_layer = input_layer
        self.dimension = emb_dimension
        if not freeze:
            self.W = torch.nn.Parameter(torch.randn(self.input_layer))
            self.W.requires_grad = True
        else:
            self.W = torch.nn.Parameter(torch.tensor([1/3, 1/3, 1/3]))
            self.W.requires_grad = False

    def forward(self, x):
        print(self.W)
        return torch.sum(x * self.W[:, None, None], axis=1)