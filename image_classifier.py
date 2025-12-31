from torch import nn
import torch.nn.functional as F
from defaults import TRAINING_CONFIG, DEVICE

default_device = DEVICE
default_dropout = TRAINING_CONFIG['dropout']

class ImageClassifier(nn.Module):
    """Image classifier to replace the classifier layer of a pretrained model in model fine-tuning."""
    
    def __init__(self, input_size, output_size, hidden_layers, device=default_device, dropout=default_dropout):
        super().__init__()

        self.layers = []

        if len(hidden_layers):
            # Create first layer: input → first hidden layer
            self.layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0], device=device)])
            # Create remaining hidden layers
            layer_pairs = zip(hidden_layers[:-1], hidden_layers[1:])
            for inp, out in layer_pairs:
                self.layers.extend([nn.Linear(inp, out, device=device)])

        out_layer_input = hidden_layers[-1] if len(hidden_layers) else input_size
        self.out = nn.Linear(out_layer_input, output_size, device=device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h):
        for layer in self.layers:
            h = self.dropout(F.relu(layer(h)))

        h = F.log_softmax(self.out(h), dim=1)

        return h
