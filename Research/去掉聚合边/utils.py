import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
import numpy as np
import pandas as pd
from torch_geometric.nn import GATv2Conv, HeteroConv
import torch.nn.functional as F
# ==============================================================================
#      【保留】您原始的 'create_temporal_edges_with_attr'
# ==============================================================================
# (这个函数是必需的，您应该已经有了)
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
#      【新增】图构建函数 (无融合边版本)
# ==============================================================================
def create_graph_list_from_df_NO_FUSES(df, wifi_cols, imu_cols, windows_size, future_steps, device):
    """
    【消融试验修改】
    1. 只创建时间边 (Temporal Edges)。
    2. 完全移除融合边 (Fuses Edges)。
    """
    graph_list = []
    num_samples = len(df) - windows_size - future_steps + 1

    for i in range(num_samples):
        window_df = df.iloc[i : i + windows_size]
        future_df = df.iloc[i + windows_size : i + windows_size + future_steps]
        
        data = HeteroData()
        
        data['wifi'].x = torch.tensor(window_df[wifi_cols].values, dtype=torch.float, device=device)
        data['imu'].x = torch.tensor(window_df[imu_cols].values, dtype=torch.float, device=device)
        
        data['wifi'].time_step = torch.arange(windows_size, device=device)
        data['imu'].time_step = torch.arange(windows_size, device=device)

        # 时间边 (Temporal Edges) - 只保留这部分
        wifi_edge_index, wifi_edge_attr = create_temporal_edges_with_attr(data['wifi'].x, device)
        imu_edge_index, imu_edge_attr = create_temporal_edges_with_attr(data['imu'].x, device)
        data[('wifi', 'temporal', 'wifi')].edge_index = wifi_edge_index
        data[('imu', 'temporal', 'imu')].edge_index = imu_edge_index
        data[('wifi', 'temporal', 'wifi')].edge_attr = wifi_edge_attr
        data[('imu', 'temporal', 'imu')].edge_attr = imu_edge_attr

        # 【核心修改】
        # --- 所有关于 'fuses' 边的创建 (create_asymmetric_fuses_edges)
        # --- 和属性分配的代码被完全删除。
        
        # 标签赋值
        labels = future_df[['x_coord', 'y_coord']].values
        data.y = torch.tensor(labels, dtype=torch.float, device=device)
        
        graph_list.append(data)
        
    return graph_list

class GNNEncoder(nn.Module):
    def __init__(self, wifi_input_dim, imu_input_dim, hidden_dim, windows_size, num_layers=2, heads=4, dropout=0.2):
        super().__init__()
        self.wifi_proj = nn.Linear(wifi_input_dim, hidden_dim)
        self.imu_proj = nn.Linear(imu_input_dim, hidden_dim)
        self.positional_embeddings = nn.Embedding(windows_size + 10, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            # 【保留】我们保留原始定义，包含 'fuses'
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
            
            # 【关键】
            # PyG 会智能地检查传入的 edge_index_dict
            # 因为我们稍后传入的数据中没有 'fuses' 边的 key，
            # GATv2Conv 的 'fuses' 部分将自动被跳过，不会报错。
            x_dict_after_conv = self.convs[i](x_dict, edge_index_dict, edge_attr_dict)
            
            for node_type, x in x_dict_after_conv.items():
                x_processed = F.gelu(x)
                x_processed = F.dropout(x_processed, p=self.dropout_p, training=self.training)
                x_dict[node_type] = self.norms[i][node_type](x_processed + residual_dict[node_type])
        
        return x_dict
    
class GNN_LSTKAN_with_Attention_v2(nn.Module):
    """
    一个包装器模块，用于按顺序运行 GNN 编码器和 预测器 (KAN 或 MLP)。
    """
    def __init__(self, gnn_encoder, kan_predictor, future_steps):
        super().__init__()
        self.gnn_encoder = gnn_encoder
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
            x_imu = x_dict['imu'].view(batch_size, window_size, hidden_dim)
            # 3. 融合多种模态 (假设：平均法)
            x_combined = (x_wifi + x_imu) / 2.0
        else:
            # 如果没有 'imu' 节点 (例如在“无融合边”实验中 GNN 未处理它)
            x_combined = x_wifi
        
        # 4. 运行预测器 (KAN 或 MLP)
        # raw_output 的形状是 [B, W, Output_Dim]，即 [128, 20, 6]
        raw_output = self.kan_predictor(x_combined) 
        
        # --- 【核心修正】---
        # 您的 KAN 返回了所有时间步的预测 [128, 20, 6]。
        # 我们只取序列的最后一个时间步的输出，形状变为 [128, 6]。
        final_prediction = raw_output[:, -1, :]
        # --- 【修正结束】---

        # 5. Reshape 最终输出
        # 现在 128*6 可以被 view 为 128*3*2
        main_pred = final_prediction.view(batch_size, self.future_steps, 2) 
        
        return main_pred, None, None

    def regularization_loss(self):
        """
        将正则化损失的调用传递给内部的预测器。
        """
        if hasattr(self.kan_predictor, 'regularization_loss'):
            return self.kan_predictor.regularization_loss()
        
        return torch.tensor(0.0, device=next(self.parameters()).device)