import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, HeteroConv, LayerNorm
import torch.nn.functional as F

class GNNEncoder(nn.Module):
    def __init__(self, wifi_input_dim, imu_input_dim, hidden_dim, windows_size, num_layers=2, heads=4, dropout=0.2):
        super().__init__()
        
        # 1. 【升级】特征投影升级为 MLP (Feature Extraction MLP)
        # 相比单层 Linear，MLP 能更好地拟合 RSSI 的非线性特征和 IMU 的复杂模式
        self.wifi_proj = nn.Sequential(
            nn.Linear(wifi_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.imu_proj = nn.Sequential(
            nn.Linear(imu_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 位置编码
        self.positional_embeddings = nn.Embedding(windows_size + 10, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # 2. GNN 层构建
        for _ in range(num_layers):
            conv = HeteroConv({
                # 时间边：捕捉自身的时间演变
                ('wifi', 'temporal', 'wifi'): GATv2Conv(-1, hidden_dim // heads, heads=heads, dropout=dropout, add_self_loops=True, edge_dim=1),
                ('imu', 'temporal', 'imu'):   GATv2Conv(-1, hidden_dim // heads, heads=heads, dropout=dropout, add_self_loops=True, edge_dim=1),
                
                # 融合边：捕捉模态间的交互 (Cross-Modal)
                # IMU -> WiFi (运动影响信号接收)
                ('imu', 'fuses', 'wifi'):     GATv2Conv((-1, -1), hidden_dim // heads, heads=heads, dropout=dropout, add_self_loops=False, edge_dim=2),
                # WiFi -> IMU (位置约束运动状态)
                ('wifi', 'fuses', 'imu'):     GATv2Conv((-1, -1), hidden_dim // heads, heads=heads, dropout=dropout, add_self_loops=False, edge_dim=1),
            }, aggr='sum')
            
            self.convs.append(conv)
            
            # 使用 PyG 自带的 Hetero LayerNorm，处理起来更方便
            self.norms.append(nn.ModuleDict({
                'wifi': nn.LayerNorm(hidden_dim),
                'imu':  nn.LayerNorm(hidden_dim),
            }))
        
        self.dropout_p = dropout

    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict

        # 1. 特征投影 (MLP)
        x_dict['wifi'] = self.wifi_proj(x_dict['wifi'])
        x_dict['imu']  = self.imu_proj(x_dict['imu'])

        # 2. 添加位置编码
        # 注意：这里假设 time_step 是从 0 到 window_size 的索引
        x_dict['wifi'] = x_dict['wifi'] + self.positional_embeddings(data['wifi'].time_step)
        x_dict['imu']  = x_dict['imu']  + self.positional_embeddings(data['imu'].time_step)

        # 3. GNN 消息传递
        for i in range(len(self.convs)):
            # 保存残差
            residual_dict = {key: x.clone() for key, x in x_dict.items()}
            
            # Convolution
            x_dict = self.convs[i](x_dict, edge_index_dict, edge_attr_dict)
            
            # Post-Processing: Act -> Dropout -> Residual -> Norm
            for node_type, x in x_dict.items():
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
                # Residual Connection
                x = x + residual_dict[node_type] 
                # Normalization
                x_dict[node_type] = self.norms[i][node_type](x)
        
        return x_dict