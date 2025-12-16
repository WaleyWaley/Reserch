import torch
import torch.nn as nn
import torch.nn.functional as F
# 假設 GNNEncoder 和 KAN 已在其他文件中定義
from GNNEncoder import GNNEncoder
from KAN_improve import KAN
class GatedMultimodalFusion(nn.Module):
    """
    【精度提升关键】门控多模态融合
    不直接相加 WiFi 和 IMU，而是通过一个 Learnable Gate 动态决定权重。
    公式: Z = z * WiFi + (1-z) * IMU
    其中 z = Sigmoid(Linear([WiFi, IMU]))
    """
    def __init__(self, feature_dim):
        super().__init__()
        # 输入是 WiFi 和 IMU 拼接后的维度
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1), # 输出一个 0~1 之间的系数
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, wifi_feat, imu_feat):
        # wifi_feat, imu_feat: [Batch, Seq, Dim]
        combined = torch.cat([wifi_feat, imu_feat], dim=-1)
        z = self.gate_net(combined) # [Batch, Seq, 1]
        
        # 动态加权融合
        fused = z * wifi_feat + (1 - z) * imu_feat
        return self.norm(fused)
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # 為了穩定性，我們可以使用一個更簡單的線性層或一個淺層 KAN
        self.attn = KAN(layers_hidden=[hidden_dim * 2, hidden_dim], use_lstm=False, dropout_rate=0, grid_size=3, spline_order=3)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, query, keys_values):
        batch_size = query.shape[0]
        seq_len = keys_values.shape[1]
        
        query_repeated = query.unsqueeze(1).repeat(1, seq_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat((query_repeated, keys_values), dim=2)))
        
        energy = energy.permute(0, 2, 1)
        v_unsq = self.v.repeat(batch_size, 1).unsqueeze(1)
        
        attention_scores = torch.bmm(v_unsq, energy).squeeze(1)
        
        return F.softmax(attention_scores, dim=1)
    
class GNN_LSTKAN_with_Attention_v3(nn.Module):
    def __init__(self, gnn_encoder, kan_predictor, future_steps):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.kan_predictor = kan_predictor
        self.future_steps = future_steps
        
        # ======================= 【修复点开始】 =======================
        # 自动判断 wifi_proj 是 Linear 还是 Sequential
        if isinstance(gnn_encoder.wifi_proj, nn.Sequential):
            # 如果是 Sequential，取最后一层 (Linear) 的输出维度
            gnn_hidden_dim = gnn_encoder.wifi_proj[-1].out_features
        else:
            # 如果是 Linear，直接取属性
            gnn_hidden_dim = gnn_encoder.wifi_proj.out_features
            
        # 获取 KAN 的输出维度 (KANLinear 有 out_features 属性，通常没问题)
        kan_output_dim = kan_predictor.kan_layers[-1].out_features
        # ======================= 【修复点结束】 =======================
        
        # 1. 门控融合模块
        self.fusion_layer = GatedMultimodalFusion(gnn_hidden_dim)
        
        # 2. 注意力模块
        self.attention = Attention(kan_output_dim)
        self.norm_attn = nn.LayerNorm(kan_output_dim) 
        
        # 3. 主预测头
        self.main_prediction_head = nn.Sequential(
            nn.Linear(kan_output_dim, kan_output_dim // 2),
            nn.SiLU(), 
            nn.Linear(kan_output_dim // 2, future_steps * 2)
        )

        # 4. 辅助预测头
        self.wifi_only_head = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim // 2, future_steps * 2)
        )
        self.imu_only_head = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim // 2, future_steps * 2)
        )

    def forward(self, data, update_grid=False):
        # 1. GNN 提取特征
        x_dict = self.gnn_encoder(data)
        num_graphs = data.num_graphs
        window_size = data['wifi'].x.size(0) // num_graphs

        wifi_feats_seq = x_dict['wifi'].view(num_graphs, window_size, -1)
        imu_feats_seq = x_dict['imu'].view(num_graphs, window_size, -1)

        # 2. 门控融合
        encoder_outputs_seq = self.fusion_layer(wifi_feats_seq, imu_feats_seq)
        
        # 3. Bi-LSTKAN
        kan_output_seq = self.kan_predictor(encoder_outputs_seq, update_grid=update_grid)
        
        # 4. Attention + Residual
        query = kan_output_seq[:, -1, :]
        attn_weights = self.attention(query, kan_output_seq)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), kan_output_seq).squeeze(1)
        
        final_vector = self.norm_attn(context_vector + kan_output_seq[:, -1, :])

        # 5. 预测
        main_prediction = self.main_prediction_head(final_vector)
        
        # 辅助路径
        wifi_last_feature = wifi_feats_seq[:, -1, :]
        imu_last_feature = imu_feats_seq[:, -1, :]
        
        wifi_only_prediction = self.wifi_only_head(wifi_last_feature)
        imu_only_prediction = self.imu_only_head(imu_last_feature)
        
        return main_prediction.view(num_graphs, self.future_steps, 2), \
               wifi_only_prediction.view(num_graphs, self.future_steps, 2), \
               imu_only_prediction.view(num_graphs, self.future_steps, 2)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        main_kan_loss = self.kan_predictor.regularization_loss(regularize_activation, regularize_entropy)
        attn_kan_loss = self.attention.attn.regularization_loss(regularize_activation, regularize_entropy)
        return main_kan_loss + attn_kan_loss