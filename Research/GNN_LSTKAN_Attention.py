import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv
from KAN_improve import KAN
from GNNEncoder import GNNEncoder

# ==============================================================================
#  【核心新增】注意力機制模塊
# ==============================================================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # 一個線性層，用於計算 Query 和 Keys 之間的“能量”或“相關性”
        # self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn = KAN(layers_hidden=[hidden_dim * 2, hidden_dim], use_lstm=False,dropout_rate=0,grid_size=3,spline_order=3)
        # 用於將能量分數轉換為最終的注意力權重的向量
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, query, keys_values):
        """
        參數:
            query (torch.Tensor): 查詢向量，形狀為 [B, H]。在我們這裡就是 LSTKAN 的最後一個隱藏狀態。
            keys_values (torch.Tensor): 鍵和值序列，形狀為 [B, T, H]。在我們這裡就是 LSTKAN 的完整輸出序列。
        """
        batch_size = query.shape[0]
        seq_len = keys_values.shape[1]
        
        # 1. 將 Query 重複 T 次，以便與 Keys 進行逐元素的比較
        query_repeated = query.unsqueeze(1).repeat(1, seq_len, 1)
        
        # 2. 計算能量分數
        # 將每個時間步的 Key 與 Query 拼接，然後通過一個全連接層和 tanh 激活
        energy = torch.tanh(self.attn(torch.cat((query_repeated, keys_values), dim=2))) # [B, T, H]
        
        # 3. 計算最終的注意力權重
        energy = energy.permute(0, 2, 1) # [B, H, T]
        v_unsq = self.v.repeat(batch_size, 1).unsqueeze(1) # [B, 1, H]
        
        # (B, 1, H) @ (B, H, T) -> (B, 1, T)
        attention_scores = torch.bmm(v_unsq, energy).squeeze(1) # [B, T]
        
        # 4. 應用 Softmax 得到權重分佈
        return F.softmax(attention_scores, dim=1)


# ==============================================================================
#  【核心升級】集成了注意力機制的端到端模型
# ==============================================================================
class GNN_LSTKAN_with_Attention(nn.Module):
    def __init__(self, gnn_encoder, kan_predictor, future_steps):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.kan_predictor = kan_predictor
        self.future_steps = future_steps
        
        kan_output_dim = kan_predictor.kan_layers[-1].out_features
        
        # 實例化注意力模塊
        self.attention = Attention(kan_output_dim)
        
        # 最終的預測頭，現在它的輸入是注意力機制的輸出
        self.prediction_head = nn.Linear(kan_output_dim, future_steps * 2)

    def forward(self, data, update_grid=False):
        # 1. GNN 提取特徵序列
        x_dict = self.gnn_encoder(data)
        num_graphs = data.num_graphs
        window_size = data['wifi'].x.size(0) // num_graphs

        wifi_feats_seq = x_dict['wifi'].view(num_graphs, window_size, -1)
        imu_feats_seq = x_dict['imu'].view(num_graphs, window_size, -1)

        encoder_outputs_seq = wifi_feats_seq + imu_feats_seq

        # 2. LSTKAN 處理序列，得到每個時間步的輸出
        kan_output_seq = self.kan_predictor(encoder_outputs_seq, update_grid=update_grid)
        
        # 3. 【新】應用注意力機制
        # a. 查詢 Query = 最後一刻的狀態
        query = kan_output_seq[:, -1, :]

        # b. 鍵/值 Keys/Values = 完整的歷史序列
        # 計算每個歷史時間步的“重要性”
        attn_weights = self.attention(query, kan_output_seq) # [B, T]
        
        # 4. 【新】計算上下文向量 (Context Vector)
        # 根據“重要性”對歷史序列進行加權求和
        attn_weights_unsq = attn_weights.unsqueeze(1) # [B, 1, T]
        # (B, 1, T) @ (B, T, H) -> (B, 1, H)
        context_vector = torch.bmm(attn_weights_unsq, kan_output_seq).squeeze(1) # [B, H]

        # 5. 使用這個高度濃縮的“上下文向量”進行最終預測
        outputs_flat = self.prediction_head(context_vector)
        return outputs_flat.view(num_graphs, self.future_steps, 2)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return self.kan_predictor.regularization_loss(regularize_activation, regularize_entropy)
