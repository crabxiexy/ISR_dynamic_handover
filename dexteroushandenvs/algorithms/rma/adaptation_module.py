
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptationModule(nn.Module):
    def __init__(self, input_dim, history_len, output_dim, hidden_dims=[32, 32, 32]):
        super(AdaptationModule, self).__init__()
        
        self.input_dim = input_dim
        self.history_len = history_len
        self.output_dim = output_dim
        
        # Architecture based on RMA paper (simplified or as appropriate)
        # Input shape: (Batch, InputDim * HistoryLen) or (Batch, InputDim, HistoryLen)
        # Common implementation uses MLP on flattened history or Conv1D
        
        # Here we use Conv1D as it handles temporal data well
        # Input to Conv1d: (Batch, InputDim, HistoryLen)
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),
            nn.ReLU()
        )
        
        # Compute flatten size
        # Assuming history_len=50
        # L_out = (L_in - kernel + 2*padding)/stride + 1
        # L1 = (50 - 8)/4 + 1 = 10.5 -> 10 (floor)
        # L2 = (10 - 5)/1 + 1 = 6
        # L3 = (6 - 5)/1 + 1 = 2
        # Final flat size = 32 * 2 = 64 (if history_len is 50)
        
        self.dummy_input = torch.zeros(1, input_dim, history_len)
        with torch.no_grad():
            self.dummy_output = self.conv_layers(self.dummy_input)
            self.flatten_dim = self.dummy_output.view(1, -1).shape[1]
            
        self.mlp = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        # x shape: (Batch, InputDim, HistoryLen)
        # If x comes in as (Batch, HistoryLen, InputDim), generic transpose needed
        if x.shape[-1] != self.history_len:
             x = x.permute(0, 2, 1) # Assuming input is (Batch, HistoryLen, InputDim)
             
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        out = self.mlp(out)
        return out
