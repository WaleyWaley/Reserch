import torch
import torch.nn.functional as F

# ==============================================================================
#  【核心新增】基於幾何學的航向變化損失函數
# ==============================================================================
def heading_change_loss(pred_seq, true_seq, epsilon=1e-8):
    """
    計算並懲罰預測軌跡與真實軌跡在航向變化（轉向角度）上的差異。

    參數:
        pred_seq (torch.Tensor): 預測的軌跡序列，形狀為 [B, S, 2]。
        true_seq (torch.Tensor): 真實的軌跡序列，形狀為 [B, S, 2]。
        epsilon (float): 一個極小值，用於防止除以零。
        
    返回:
        torch.Tensor: 一個標量的損失值。
    """
    # 至少需要3個點才能計算一次航向變化
    if pred_seq.shape[1] < 3:
        return torch.tensor(0.0, device=pred_seq.device)

    # 1. 計算速度向量 (t - (t-1))
    pred_vel = pred_seq[:, 1:] - pred_seq[:, :-1]
    true_vel = true_seq[:, 1:] - true_seq[:, :-1]
    
    # 2. 將速度向量歸一化為單位方向向量
    # F.normalize 會處理長度為0的向量，避免除以零
    pred_direction = F.normalize(pred_vel, p=2, dim=-1, eps=epsilon)
    true_direction = F.normalize(true_vel, p=2, dim=-1, eps=epsilon)

    # 3. 計算連續方向向量之間的點積
    # 點積的值等於 cos(theta)，其中 theta 是兩個向量間的夾角
    pred_dot_products = (pred_direction[:, 1:] * pred_direction[:, :-1]).sum(dim=-1)
    true_dot_products = (true_direction[:, 1:] * true_direction[:, :-1]).sum(dim=-1)
    
    # 4. 計算角度損失
    # 我們可以直接比較 cos(theta) 的差異，這在數值上比計算 acos 更穩定
    # MSE(cos(theta_pred), cos(theta_true))
    angle_loss = F.mse_loss(pred_dot_products, true_dot_products)
    
    return angle_loss

# --- 我們之前的損失函數 (保持不變) ---

def trajectory_dynamics_loss(pred_seq, true_seq):
    if pred_seq.shape[1] < 3:
        return torch.tensor(0.0, device=pred_seq.device)
    pred_vel = pred_seq[:, 1:] - pred_seq[:, :-1]
    true_vel = true_seq[:, 1:] - true_seq[:, :-1]
    pred_accel = pred_vel[:, 1:] - pred_vel[:, :-1]
    true_accel = true_vel[:, 1:] - true_vel[:, :-1]
    velocity_loss = F.mse_loss(pred_vel, true_vel)
    acceleration_loss = F.mse_loss(pred_accel, true_accel)
    return velocity_loss + 2.0 * acceleration_loss

def physics_violation_loss(pred_seq, dt=0.05, max_speed=3.0, max_accel=5.0):
    if pred_seq.shape[1] < 2:
        return torch.tensor(0.0, device=pred_seq.device)
    pred_vel = (pred_seq[:, 1:] - pred_seq[:, :-1]) / dt
    pred_speed = torch.linalg.norm(pred_vel, dim=-1)
    if pred_seq.shape[1] < 3:
        pred_accel_magnitude = torch.tensor(0.0, device=pred_seq.device)
    else:
        pred_accel = (pred_vel[:, 1:] - pred_vel[:, :-1]) / dt
        pred_accel_magnitude = torch.linalg.norm(pred_accel, dim=-1)
    speed_violation = F.relu(pred_speed - max_speed)
    accel_violation = F.relu(pred_accel_magnitude - max_accel)
    loss = torch.mean(speed_violation) + torch.mean(accel_violation)
    return loss