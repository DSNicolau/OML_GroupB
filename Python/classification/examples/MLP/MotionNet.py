import torch
import torch.nn as nn

activations = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh()
}

class MotionNet(nn.Module):
    def __init__(self, input_size=1, hidden_layers=2, hidden_units=64, dropout_rate=0, activation="relu"):
        super(MotionNet, self).__init__()
        if hidden_layers < 0:
            raise ValueError('hidden_layers must be >= 0')
        
        self.hidden_layers = hidden_layers
        
        self.in_linear = nn.Linear(input_size, hidden_units)
        self.in_activ = nn.ReLU()
        
        for i in range(hidden_layers):
            self.add_module("linear{}".format(i+1), nn.Linear(hidden_units, hidden_units))
            self.add_module("activ{}".format(i+1), activations[activation])
            self.add_module("dropout{}".format(i+1), nn.Dropout(dropout_rate))
                
        self.out_linear = nn.Linear(hidden_units, 2) 
        self.out_activ = activations[activation]
        self.softmax = nn.Softmax()
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=1.0)
    #         if module.bias is not None:
    #             module.bias.data.zero_()   
    def forward(self, x):
        out = self.in_linear(x)
        out = self.in_activ(out)
        for i in range(self.hidden_layers):
            out = self._modules["linear{}".format(i+1)](out)
            out = self._modules["activ{}".format(i+1)](out)
            out = self._modules["dropout{}".format(i+1)](out)
        out = self.out_linear(out)
        out = self.out_activ(out)
        return out