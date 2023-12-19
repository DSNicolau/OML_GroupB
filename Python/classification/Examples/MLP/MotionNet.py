import torch
import torch.nn as nn

class MotionNet(nn.Module):
    def __init__(self, input_size=1, hidden_units=64, dropout_rate=0):
        super(MotionNet, self).__init__()
        self.linear = nn.Linear(input_size, hidden_units)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_units, hidden_units) 
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.linear3 = nn.Linear(hidden_units, hidden_units) 
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.linear4 = nn.Linear(hidden_units, 2) 
        self.relu4 = nn.ReLU()
        self.softmax = nn.Softmax()
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=1.0)
    #         if module.bias is not None:
    #             module.bias.data.zero_()   
    def forward(self, x):
        out = self.linear(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        out = self.relu3(out)
        out = self.linear4(out)
        out = self.relu4(out)
        out = self.softmax(out)
        return out