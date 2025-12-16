import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, HeteroConv
import torch.nn.functional as F

class GNNEncoder(nn.Module):
    def __init__(self, wifi_input_dim, imu_input_dim, hidden_dim, windows_size, num_layers=2, heads=4, dropout=0.2):
        super().__init__()
        self.wifi_proj = nn.Linear(wifi_input_dim, hidden_dim)
        self.imu_proj = nn.Linear(imu_input_dim, hidden_dim)
        self.positional_embeddings = nn.Embedding(windows_size + 10, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv({
                ('wifi', 'temporal', 'wifi'): GATv2Conv(-1, hidden_dim // heads, heads=heads, dropout=dropout, add_self_loops=False, edge_dim=1),
                ('imu', 'temporal', 'imu'): GATv2Conv(-1, hidden_dim // heads, heads=heads, dropout=dropout, add_self_loops=False, edge_dim=1),
                ('imu', 'fuses', 'wifi'): GATv2Conv((-1, -1), hidden_dim // heads, heads=heads, dropout=dropout, add_self_loops=False, edge_dim=2),
                ('wifi', 'fuses', 'imu'): GATv2Conv((-1, -1), hidden_dim // heads, heads=heads, dropout=dropout, add_self_loops=False, edge_dim=1),
            }, aggr='sum')
            self.convs.append(conv)
            self.norms.append(nn.ModuleDict({
                'wifi': nn.LayerNorm(hidden_dim),
                'imu': nn.LayerNorm(hidden_dim),
            }))
        
        self.dropout_p = dropout

    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict

        x_dict['wifi'] = self.wifi_proj(x_dict['wifi'])
        x_dict['imu'] = self.imu_proj(x_dict['imu'])
        x_dict['wifi'] = x_dict['wifi'] + self.positional_embeddings(data['wifi'].time_step)
        x_dict['imu'] = x_dict['imu'] + self.positional_embeddings(data['imu'].time_step)

        for i in range(len(self.convs)):
            residual_dict = {key: x.clone() for key, x in x_dict.items()}
            x_dict_after_conv = self.convs[i](x_dict, edge_index_dict, edge_attr_dict)
            for node_type, x in x_dict_after_conv.items():
                x_processed = F.gelu(x)
                x_processed = F.dropout(x_processed, p=self.dropout_p, training=self.training)
                x_dict[node_type] = self.norms[i][node_type](x_processed + residual_dict[node_type])
        
        return x_dict