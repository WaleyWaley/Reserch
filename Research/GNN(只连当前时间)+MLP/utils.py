import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import numpy as np
import pandas as pd

# ==============================================================================
#                      GNNEncoder (来自您的代码，保持不变)
# ==============================================================================
from torch_geometric.nn import GATv2Conv, HeteroConv

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

# ==============================================================================
#                  create_temporal_edges_with_attr (来自您的代码)
# ==============================================================================
def create_temporal_edges_with_attr(node_features, device):
    """辅助函数：根据时间戳创建时间边和delta_t边特征"""
    num_nodes = node_features.size(0)
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long, device=device), \
               torch.empty((0, 1), dtype=torch.float, device=device)

    source_nodes = torch.arange(0, num_nodes - 1, device=device)
    dest_nodes = torch.arange(1, num_nodes, device=device)
    edge_index = torch.stack([source_nodes, dest_nodes], dim=0)
    
    feature_diff = node_features[1:] - node_features[:-1]
    edge_attr = torch.linalg.norm(feature_diff, dim=1).view(-1, 1)

    return edge_index, edge_attr

# ==============================================================================
#          【全新】create_graph_list_from_df (消融试验版)
# ==============================================================================
def create_graph_list_from_df_ablation(df, wifi_cols, imu_cols, windows_size, future_steps, device):
    """
    【消融试验修改】
    1. 移除 past_radius 和 future_radius 参数。
    2. 融合边 (Fuses Edges) 只在当前时刻 i <-> i 连接。
    """
    graph_list = []
    num_samples = len(df) - windows_size - future_steps + 1

    accel_cols = [col for col in imu_cols if 'accelerometer' in col]
    gyro_cols = [col for col in imu_cols if 'gyroscope' in col]
    rss_cols = [col for col in wifi_cols if 'RSSI' in col]

    # 【消融试验修改】
    # 预先创建当前时刻的连接 (i -> i)
    # 这是一个 [0, 1, ..., N-1] 的索引
    identity_indices = torch.arange(windows_size, dtype=torch.long, device=device)
    # 边索引为 [[0, 1, ...], [0, 1, ...]]
    edge_index_current = torch.stack([identity_indices, identity_indices], dim=0)


    for i in range(num_samples):
        window_df = df.iloc[i : i + windows_size]
        future_df = df.iloc[i + windows_size : i + windows_size + future_steps]
        
        data = HeteroData()
        
        data['wifi'].x = torch.tensor(window_df[wifi_cols].values, dtype=torch.float, device=device)
        data['imu'].x = torch.tensor(window_df[imu_cols].values, dtype=torch.float, device=device)
        
        data['wifi'].time_step = torch.arange(windows_size, device=device)
        data['imu'].time_step = torch.arange(windows_size, device=device)

        # 时间边 (Temporal Edges) - 保持不变
        wifi_edge_index, wifi_edge_attr = create_temporal_edges_with_attr(data['wifi'].x, device)
        imu_edge_index, imu_edge_attr = create_temporal_edges_with_attr(data['imu'].x, device)
        data[('wifi', 'temporal', 'wifi')].edge_index = wifi_edge_index
        data[('imu', 'temporal', 'imu')].edge_index = imu_edge_index
        data[('wifi', 'temporal', 'wifi')].edge_attr = wifi_edge_attr
        data[('imu', 'temporal', 'imu')].edge_attr = imu_edge_attr

        # 融合边 (Fuses Edges) - 【消融试验修改】
        # 分配 i <-> i 的边
        data[('imu', 'fuses', 'wifi')].edge_index = edge_index_current.clone()
        data[('wifi', 'fuses', 'imu')].edge_index = edge_index_current.clone()
        
        # --- 边属性的分配逻辑 ---
        # 属性总是来自于源 (source) 节点
        
        # for imu -> fuses -> wifi
        # 源索引是 [0, 1, ..., N-1]
        forward_src_indices = edge_index_current[0] 
        motion_intensity = np.linalg.norm(window_df[accel_cols].values, axis=1)
        motion_intensity_tensor = torch.tensor(motion_intensity, dtype=torch.float, device=device)
        rotation_intensity = np.linalg.norm(window_df[gyro_cols].values, axis=1)
        rotation_intensity_tensor = torch.tensor(rotation_intensity, dtype=torch.float, device=device)
        
        fuses_edge_attr_forward = torch.cat([
            motion_intensity_tensor[forward_src_indices].view(-1, 1), 
            rotation_intensity_tensor[forward_src_indices].view(-1, 1)
        ], dim=1)
        data[('imu', 'fuses', 'wifi')].edge_attr = fuses_edge_attr_forward

        # for wifi -> fuses -> imu
        # 源索引是 [0, 1, ..., N-1]
        backward_src_indices = edge_index_current[0] 
        avg_rssi_per_node = window_df[rss_cols].mean(axis=1).values
        avg_rssi_tensor = torch.tensor(avg_rssi_per_node, dtype=torch.float, device=device)
        fuses_edge_attr_backward = avg_rssi_tensor[backward_src_indices].view(-1, 1)
        data[('wifi', 'fuses', 'imu')].edge_attr = fuses_edge_attr_backward

        # 标签赋值
        labels = future_df[['x_coord', 'y_coord']].values
        data.y = torch.tensor(labels, dtype=torch.float, device=device)
        
        graph_list.append(data)
        
    return graph_list

# ==============================================================================
#                      【全新】MLP 预测器 (替换 KAN)
# ==============================================================================
class MLPPredictor(nn.Module):
    def __init__(self, layers_hidden, use_lstm=True, lstm_hidden=32, dropout_rate=0.1):
        """
        MLP 预测器，可选择性地在开头使用 LSTM。
        layers_hidden: 像 [input_dim, hidden1, hidden2, ..., output_dim]
        """
        super().__init__()
        self.use_lstm = use_lstm
        
        # 决定 MLP 的输入维度
        mlp_input_dim = layers_hidden[0] # GNN hidden dim
        
        if use_lstm:
            self.lstm = nn.LSTM(input_size=layers_hidden[0],
                                hidden_size=lstm_hidden,
                                num_layers=1,
                                batch_first=True)
            # 如果使用LSTM，MLP的输入维度变为LSTM的隐藏层维度
            mlp_input_dim = lstm_hidden
        else:
            self.lstm = None
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 构建 MLP
        layers = []
        # MLP 的层级是 [mlp_input_dim, hidden1, ..., output_dim]
        # layers_hidden[1:-1] 是 [hidden1, hidden2, ...]
        current_dim = mlp_input_dim
        for hidden_dim in layers_hidden[1:-1]:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU()) # 使用GELU激活
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        # 添加最后一层 (从 current_dim 到 output_dim)
        layers.append(nn.Linear(current_dim, layers_hidden[-1]))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x 形状: (batch_size, window_size, gnn_hidden_dim)
        
        if self.use_lstm:
            # lstm_out 形状: (batch_size, window_size, lstm_hidden)
            lstm_out, _ = self.lstm(x)
            # 我们只取序列的最后一个时间步的输出来做预测
            x = lstm_out[:, -1, :] # 形状: (batch_size, lstm_hidden)
        else:
            # 如果没有LSTM，也只取最后一个时间步的特征
            x = x[:, -1, :] # 形状: (batch_size, gnn_hidden_dim)
        
        x = self.dropout(x)
        return self.mlp(x)

    def regularization_loss(self):
        """
        【消融试验修改】
        提供一个虚拟的正则化损失函数，以确保它能无缝替换 KAN。
        MLP 本身没有正则化损失，所以返回 0。
        """
        # 确保返回一个在正确设备上的 0.0
        # 检查 self.mlp 是否至少有一个
        if len(self.mlp) > 0 and isinstance(self.mlp[0], nn.Linear):
             return torch.tensor(0.0, device=self.mlp[0].weight.device)
        return torch.tensor(0.0)

class GNN_LSTKAN_with_Attention_v2(nn.Module):
    """
    一个包装器模块，用于按顺序运行 GNN 编码器和 预测器 (KAN 或 MLP)。
    它还处理 GNN 的异构输出 ('wifi', 'imu') 到预测器所需的
    单一 (batch_size, window_size, hidden_dim) 输入张量的转换。
    """
    def __init__(self, gnn_encoder, kan_predictor, future_steps):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        # kan_predictor 参数可以接收 KAN 实例，也可以接收 MLPPredictor 实例
        self.kan_predictor = kan_predictor 
        self.future_steps = future_steps

    def forward(self, data):
        # 1. 运行 GNN 编码器
        x_dict = self.gnn_encoder(data)
        
        # 2. Reshape 节点特征
        batch_size = data.num_graphs
        
        # 假设 'wifi' 节点总是存在且数量正确
        num_wifi_nodes = x_dict['wifi'].shape[0]
        
        if num_wifi_nodes % batch_size != 0:
            raise ValueError(f"节点数量 ({num_wifi_nodes}) 与 batch_size ({batch_size}) 无法匹配。")
            
        window_size = num_wifi_nodes // batch_size
        hidden_dim = x_dict['wifi'].shape[1]
        
        x_wifi = x_dict['wifi'].view(batch_size, window_size, hidden_dim)
        
        # 检查 'imu' 节点是否存在
        if 'imu' in x_dict and x_dict['imu'].shape[0] > 0:
            # 确保 IMU 节点数量也正确
            if x_dict['imu'].shape[0] != num_wifi_nodes:
                # 在您的消融试验中 (i <-> i)，它们数量应该相同
                print(f"警告: WiFi 节点数 ({num_wifi_nodes}) 与 IMU 节点数 ({x_dict['imu'].shape[0]}) 不匹配。")
                
            x_imu = x_dict['imu'].view(batch_size, window_size, hidden_dim)
            # 3. 融合多种模态 (假设：平均法)
            x_combined = (x_wifi + x_imu) / 2.0
        else:
            # 如果没有 'imu' 节点（例如，如果GNN层只输出了'wifi'）
            x_combined = x_wifi
        
        # 4. 运行预测器 (KAN 或 MLP)
        raw_output = self.kan_predictor(x_combined)
        
        # 5. Reshape 最终输出
        main_pred = raw_output.view(batch_size, self.future_steps, 2)
        
        return main_pred, None, None

    def regularization_loss(self):
        """
        将正则化损失的调用传递给内部的预测器。
        (MLPPredictor 会返回 0，KAN 会返回其正则化损失)
        """
        if hasattr(self.kan_predictor, 'regularization_loss'):
            return self.kan_predictor.regularization_loss()
        
        # 如果预测器没有这个方法，返回 0
        return torch.tensor(0.0, device=next(self.parameters()).device)