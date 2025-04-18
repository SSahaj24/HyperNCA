import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10)


def make_sequental_mlp(input_space, action_space, hidden_dim, bias, layers): 
    
    fc_in = nn.Linear(input_space, hidden_dim, bias=bias)
    fc_out = nn.Linear(hidden_dim, action_space , bias=bias)
    tanh = torch.nn.Tanh()
    layer_list = [fc_in, tanh]
    for i in range(1, layers-1):
        layer_list.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        layer_list.append(torch.nn.Tanh())
    layer_list.append(fc_out)
    
    return torch.nn.Sequential(*layer_list)


class MLPn(nn.Module):        
    
    def __init__(self, input_space, action_space, hidden_dim, bias, layers, device=None):
        super(MLPn, self).__init__()
        # Use specified device or auto-detect
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.out = make_sequental_mlp(input_space, action_space, hidden_dim, bias, layers)
        # Move the model to the device
        self.to(self.device)

    def forward(self, x):
        # Ensure input is on the correct device
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
        else:
            x = torch.tensor(x, device=self.device, dtype=torch.float64)
        return self.out(x)