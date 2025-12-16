import torch
import torch.nn.functional as F

# ==============================================================================
#  【核心新增】基於最小加加速度原理的軌跡平滑損失
# ==============================================================================
def jerk_loss(pred_seq, dt=0.05, epsilon=1e-8):
    """
    計算預測軌跡的加加速度 (Jerk) 損失。
    這是一個無監督的損失，旨在懲罰軌跡的不平順性。

    參數:
        pred_seq (torch.Tensor): 預測的軌跡序列，形狀為 [B, S, 2]。
        dt (float): 每個時間步之間的時間間隔（秒）。
        
    返回:
        torch.Tensor: 一個標量的損失值。
    """
    # 至少需要4個點才能計算一次加加速度
    if pred_seq.shape[1] < 4:
        return torch.tensor(0.0, device=pred_seq.device)

    # 1. 計算速度向量 (m/s)
    pred_vel = (pred_seq[:, 1:] - pred_seq[:, :-1]) / dt
    
    # 2. 計算加速度向量 (m/s^2)
    pred_accel = (pred_vel[:, 1:] - pred_vel[:, :-1]) / dt

    # 3. 計算加加速度向量 (m/s^3)
    pred_jerk = (pred_accel[:, 1:] - pred_accel[:, :-1]) / dt
    
    # 4. 計算加加速度的 L2 範數的平方，並求均值
    # 我們懲罰加加速度的大小，目標是使其盡可能接近於零
    # 使用 .pow(2).mean() 作為損失，比直接用 norm 更穩定
    jerk_magnitude_sq = pred_jerk.pow(2).sum(dim=-1)
    
    return torch.mean(jerk_magnitude_sq)


# --- 我們之前的損失函數 (可以選擇性地與 JerkLoss 結合使用) ---
def heading_change_loss(pred_seq, true_seq, epsilon=1e-8):
    # ... (此函數代碼與之前相同)
    if pred_seq.shape[1] < 3:
        return torch.tensor(0.0, device=pred_seq.device)
    pred_vel = pred_seq[:, 1:] - pred_seq[:, :-1]
    true_vel = true_seq[:, 1:] - true_seq[:, :-1]
    pred_direction = F.normalize(pred_vel, p=2, dim=-1, eps=epsilon)
    true_direction = F.normalize(true_vel, p=2, dim=-1, eps=epsilon)
    pred_dot_products = (pred_direction[:, 1:] * pred_direction[:, :-1]).sum(dim=-1)
    true_dot_products = (true_direction[:, 1:] * true_direction[:, :-1]).sum(dim=-1)
    angle_loss = F.mse_loss(pred_dot_products, true_dot_products)
    return angle_loss
