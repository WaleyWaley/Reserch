import torch
import torch.nn.functional as F

def trajectory_dynamics_loss(pred_seq, true_seq):
    """
    計算軌跡的動態損失 (速度 + 加速度)，旨在讓軌跡形狀更逼真。
    
    參數:
        pred_seq (torch.Tensor): 預測的軌跡序列，形狀為 [B, S, 2]。
        true_seq (torch.Tensor): 真實的軌跡序列，形狀為 [B, S, 2]。
        
    返回:
        torch.Tensor: 一個標量的損失值。
    """
    # S (序列長度) 必須大於 2 才能計算加速度
    if pred_seq.shape[1] < 3:
        return torch.tensor(0.0, device=pred_seq.device)

    # 1. 計算速度 (t - (t-1))
    pred_vel = pred_seq[:, 1:] - pred_seq[:, :-1]
    true_vel = true_seq[:, 1:] - true_seq[:, :-1]
    
    # 2. 計算加速度 (vel_t - vel_{t-1})
    pred_accel = pred_vel[:, 1:] - pred_vel[:, :-1]
    true_accel = true_vel[:, 1:] - true_vel[:, :-1]
    
    # 3. 分別計算速度和加速度的 MSE 損失
    velocity_loss = F.mse_loss(pred_vel, true_vel)
    acceleration_loss = F.mse_loss(pred_accel, true_accel)
    
    # 4. 將兩者相加作為總的動態損失
    # 我們可以給予加速度損失更高的權重，因為它對轉彎更敏感
    return velocity_loss + 2.0 * acceleration_loss

# ==============================================================================
#  【核心新增】物理學約束損失函數
# ==============================================================================
def physics_violation_loss(pred_seq, dt=0.05, max_speed=3.0, max_accel=5.0):
    """
    計算違反物理學常識的損失。
    懲罰那些速度或加速度超過合理上限的預測。

    參數:
        pred_seq (torch.Tensor): 預測的軌跡序列，形狀為 [B, S, 2]。
        dt (float): 每個時間步之間的時間間隔（秒）。我們的數據是20Hz，所以是 1/20 = 0.05s。
        max_speed (float): 合理的最大速度上限 (米/秒)。例如，3 m/s 約等於快跑的速度。
        max_accel (float): 合理的最大加速度上限 (米/秒^2)。5 m/s^2 是一個很劇烈的啟動/制動。
        
    返回:
        torch.Tensor: 一個標量的損失值。
    """
    if pred_seq.shape[1] < 2: # 至少需要2個點才能計算速度
        return torch.tensor(0.0, device=pred_seq.device)

    # 1. 計算物理速度 (位移除以時間)
    pred_vel = (pred_seq[:, 1:] - pred_seq[:, :-1]) / dt
    pred_speed = torch.linalg.norm(pred_vel, dim=-1) # 計算速度大小 (L2 範數)

    # 2. 計算物理加速度 (速度變化除以時間)
    if pred_seq.shape[1] < 3:
        # 如果點不夠計算加速度，則加速度損失為0
        pred_accel_magnitude = torch.tensor(0.0, device=pred_seq.device)
    else:
        pred_accel = (pred_vel[:, 1:] - pred_vel[:, :-1]) / dt
        pred_accel_magnitude = torch.linalg.norm(pred_accel, dim=-1) # 計算加速度大小

    # 3. 計算超出上限的部分
    # F.relu(x - threshold) 是一個巧妙的技巧，它只保留 x > threshold 的部分
    speed_violation = F.relu(pred_speed - max_speed)
    accel_violation = F.relu(pred_accel_magnitude - max_accel)

    # 4. 對超出部分進行懲罰 (取均值)
    # 我們只懲罰那些“出格”的預測
    loss = torch.mean(speed_violation) + torch.mean(accel_violation)
    
    return loss
