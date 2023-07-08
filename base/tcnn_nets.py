import torch
import torch.nn as nn
import tinycudann as tcnn


### hashgrid
class HashGridMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.encoding = tcnn.Encoding(
            in_features,
            dtype=torch.float32,
            encoding_config={
                "otype": "Grid",
                "type": "Hash",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.5,
                "interpolation": "Linear"
            },
        )

        self.net = nn.Sequential(
            nn.Linear(self.encoding.n_output_dims, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )

    def forward(self, x):
        # x in [-1, 1]
        # assert x.min() >= -1 - 1e-3 and x.max() <= 1 + 1e-3
        x = (x + 1) / 2 # [0, 1]
        x_shape = x.shape
        output = self.encoding(x.view(-1, self.in_features))
        output = self.net(output).view(list(x_shape[:-1]) + [self.out_features])
        return output

    #     self.net = tcnn.NetworkWithInputEncoding(
    #         in_features, 
    #         out_features,
    #         encoding_config={
    #             "otype": "Grid",
    #             "type": "Hash",
    #             "n_levels": 16,
    #             "n_features_per_level": 2,
    #             "log2_hashmap_size": 19,
    #             "base_resolution": 16,
    #             "per_level_scale": 2.0,
    #             "interpolation": "Linear"
    #         },
    #         network_config={
    #             "otype": "FullyFusedMLP",
    #             "activation": "ReLU",
    #             "output_activation": "None",
    #             "n_neurons": 64,
    #             "n_hidden_layers": 1,
    #         }
    #     )
    
    # def forward(self, x):
    #     # x in [-1, 1]
    #     assert x.min() >= -1 - 1e-3 and x.max() <= 1 + 1e-3
    #     output = self.net(x.view(-1, self.in_features)).view(list(x.shape[:-1]) + [self.out_features]).to(x)
    #     return output
        

class DenseGridMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.encoding = tcnn.Encoding(
            in_features,
            dtype=torch.float32,
            encoding_config={
                "otype": "Grid",
                "type": "Dense",
                "n_levels": 1,
                "n_features_per_level": 8,
                "base_resolution": 512,
                "per_level_scale": 1.5,
                "interpolation": "Linear"
            },
        )

        self.net = nn.Sequential(
            nn.Linear(self.encoding.n_output_dims, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )

    def forward(self, x):
        # x in [-1, 1]
        # assert x.min() >= -1 - 1e-3 and x.max() <= 1 + 1e-3
        x = (x + 1) / 2 # [0, 1]
        x_shape = x.shape
        output = self.encoding(x.view(-1, self.in_features))
        output = self.net(output).view(list(x_shape[:-1]) + [self.out_features])
        return output
