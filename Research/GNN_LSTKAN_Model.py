import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, HeteroConv
import torch.nn.functional as F
from kan_improved import KAN, KANLinear
from GNNEncoder import GNNEncoder
# ==============================================================================
#  第一部分：GNN 编码器 (源自 seq2seq_model.py)
#  职责：从异构图中提取每个时间步的节点特征。
# ==============================================================================
# class GNNEncoder(nn.Module):
#     def __init__(self, wifi_input_dim, imu_input_dim, hidden_dim, windows_size, num_layers=2, heads=4, dropout=0.2):
#         super().__init__()
#         self.wifi_proj = nn.Linear(wifi_input_dim, hidden_dim)
#         self.imu_proj = nn.Linear(imu_input_dim, hidden_dim)
#         self.positional_embeddings = nn.Embedding(windows_size + 10, hidden_dim)
#         self.convs = nn.ModuleList()
#         self.norms = nn.ModuleList()

#         for _ in range(num_layers):
#             conv = HeteroConv({
#                 ('wifi', 'temporal', 'wifi'): GATv2Conv(-1, hidden_dim // heads, heads=heads, dropout=dropout, add_self_loops=False, edge_dim=1),
#                 ('imu', 'temporal', 'imu'): GATv2Conv(-1, hidden_dim // heads, heads=heads, dropout=dropout, add_self_loops=False, edge_dim=1),
#                 ('imu', 'fuses', 'wifi'): GATv2Conv((-1, -1), hidden_dim // heads, heads=heads, dropout=dropout, add_self_loops=False, edge_dim=2),
#                 ('wifi', 'fuses', 'imu'): GATv2Conv((-1, -1), hidden_dim // heads, heads=heads, dropout=dropout, add_self_loops=False, edge_dim=1),
#             }, aggr='sum')
#             self.convs.append(conv)
#             self.norms.append(nn.ModuleDict({
#                 'wifi': nn.LayerNorm(hidden_dim),
#                 'imu': nn.LayerNorm(hidden_dim),
#             }))
        
#         self.dropout_p = dropout

#     def forward(self, data):
#         x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict

#         x_dict['wifi'] = self.wifi_proj(x_dict['wifi'])
#         x_dict['imu'] = self.imu_proj(x_dict['imu'])
#         x_dict['wifi'] = x_dict['wifi'] + self.positional_embeddings(data['wifi'].time_step)
#         x_dict['imu'] = x_dict['imu'] + self.positional_embeddings(data['imu'].time_step)

#         for i in range(len(self.convs)):
#             residual_dict = {key: x.clone() for key, x in x_dict.items()}
#             x_dict_after_conv = self.convs[i](x_dict, edge_index_dict, edge_attr_dict)
#             for node_type, x in x_dict_after_conv.items():
#                 x_processed = F.gelu(x)
#                 x_processed = F.dropout(x_processed, p=self.dropout_p, training=self.training)
#                 x_dict[node_type] = self.norms[i][node_type](x_processed + residual_dict[node_type])
        
#         return x_dict

# ==============================================================================
#  第二部分：端到端的 GNN + LSTKAN 整合模型
#  这是我们将 GNN 编码器和 LSTKAN 预测器结合在一起的最终模型。
# ==============================================================================
class GNN_plus_LSTKAN(nn.Module):
    def __init__(self, gnn_encoder, kan_predictor, future_steps):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.kan_predictor = kan_predictor
        self.future_steps = future_steps
        # 获取 KAN 网络的最终输出维度
        kan_output_feature_dim = kan_predictor.kan_layers[-1].out_features

        # 创建一个最终的线性预测头
        # 它将 KAN 序列的最后一个时间步的输出，映射到最终的预测坐标
        self.prediction_head = nn.Linear(kan_output_feature_dim, future_steps * 2)

    def forward(self, data, update_grid=False):
        # 1. GNN 编码器提取节点特征
        x_dict = self.gnn_encoder(data)

        # 2. 将节点特征聚合成时间步特征序列
        num_graphs = data.num_graphs
        window_size = data['wifi'].x.size(0) // num_graphs
        
        wifi_feats_seq = x_dict['wifi'].view(num_graphs, window_size, -1)
        imu_feats_seq = x_dict['imu'].view(num_graphs, window_size, -1)
        
        # 将两种模态的特征融合（例如，相加），作为 LSTKAN 的输入
        encoder_outputs_seq = wifi_feats_seq + imu_feats_seq # 形状: [batch, window_size, hidden_dim]

        # 3. LSTKAN 模型处理整个特征序列
        # 注意：我们将 update_grid 标志传递给 KAN
        kan_output_seq = self.kan_predictor(encoder_outputs_seq, update_grid=update_grid)
        # kan_output_seq 形状: [batch, window_size, kan_last_hidden_dim]

        device = self.prediction_head.weight.device
        # 4. 提取最后一个时间步的特征进行预测
        last_time_step_features = kan_output_seq[:, -1, :].to(device)
        
        # 5. 通过最终的预测头得到结果
        outputs_flat = self.prediction_head(last_time_step_features)
        
        return outputs_flat.view(num_graphs, self.future_steps, 2)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """辅助函数，直接调用 KAN 内部的正则化损失"""
        return self.kan_predictor.regularization_loss(regularize_activation, regularize_entropy)
