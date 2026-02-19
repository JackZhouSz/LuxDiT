import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange, repeat


class HDR_MLP(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_dim=2, out_dim=1, hidden_dim=64, num_layers=4, out_activation='exp'):
        super(HDR_MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the layers
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, out_dim))

        # Register the layers as a ModuleList
        self.layers = nn.ModuleList(layers)

        # Initialize weights
        for layer in self.layers:
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

        # Set the output activation function
        if out_activation == 'exp':
            self.out_activation = lambda x: torch.exp(x.clamp(-10, 10) - 1)
        elif out_activation == 'softplus':
            self.out_activation = lambda x: F.softplus(x)


    def forward(self, x_ldr, x_hdr):
        x = torch.cat((x_ldr, x_hdr), dim=-1)
        for i in range(self.num_layers - 1):
            x = F.leaky_relu(self.layers[i](x))
        x = self.layers[-1](x)
        x = self.out_activation(x)
        return x


class HDR_CNN(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_dim=6, out_dim=3, hidden_dim=64, num_conv_layers=2, num_fc_layers=2, out_activation='exp'):
        super(HDR_CNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.num_fc_layers = num_fc_layers

        # Define the convolutional layers
        conv_layers = []
        conv_layers.append(nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1))
        for _ in range(num_conv_layers - 1):
            conv_layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))

        # Define the fully connected layers, 1x1 conv layer
        for _ in range(num_fc_layers - 1):
            conv_layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
        conv_layers.append(nn.Conv2d(hidden_dim, out_dim, kernel_size=1))
        self.conv_layers = nn.ModuleList(conv_layers)

        # Initialize weights
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

        # Set the output activation function
        if out_activation == 'exp':
            self.out_activation = lambda x: torch.exp(x.clamp(-10, 10) - 1)
        elif out_activation == 'softplus':
            self.out_activation = lambda x: F.softplus(x)

    def forward(self, x_ldr, x_hdr, format='HWC'):
        if format == 'HWC':
            x = torch.cat((x_ldr, x_hdr), dim=-1)
            x = rearrange(x, '... h w c -> ... c h w')
        elif format == 'CHW':
            x = torch.cat((x_ldr, x_hdr), dim=-3)

        out_shape = x_ldr.shape
        C, H, W = x.shape[-3:]
        if x.ndim > 4:
            x = x.reshape(-1, C, H, W)
        
        for conv_layer in self.conv_layers:
            x = F.leaky_relu(conv_layer(x))

        x = self.out_activation(x)
        if format == 'HWC':
            x = rearrange(x, '... c h w -> ... h w c')

        x = x.reshape(out_shape)
        return x

