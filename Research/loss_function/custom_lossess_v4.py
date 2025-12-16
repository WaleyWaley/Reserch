import torch
import torch.nn.functional as F

"""
==============================================================================
            改進版的自定義損失函數 (版本 2)
==============================================================================
核心改進思想：
1.  從懲罰「加速度」升級為懲罰「加加速度 (Jerk)」，物理意義更合理。
2.  對損失項進行對數縮放 (Log Scaling) 和裁剪 (Clamping)，增強數值穩定性。
3.  確保損失值始終為正，且不會因極端輸入而產生梯度爆炸。
==============================================================================
"""

# ==============================================================================
#  1. 【改進】躍度平滑性損失 (Jerk Smoothness Loss)
# ==============================================================================
def jerk_smoothness_loss_v2(pred_seq, dt=0.05, epsilon=1e-8):
    """
    懲罰軌跡的躍度 (Jerk, 加速度的變化率)，鼓勵加速度本身是平滑的。
    這比直接懲罰加速度更合理，因為它允許模型進行加速運動，但要求加速過程是平滑的。
    """
    # 至少需要4個點才能計算躍度 (pos -> vel -> accel -> jerk)
    if pred_seq.shape[1] < 4:
        return torch.tensor(0.0, device=pred_seq.device, dtype=pred_seq.dtype)

    # 計算速度 (m/s)
    pred_vel = (pred_seq[:, 1:] - pred_seq[:, :-1]) / dt
    
    # 計算加速度 (m/s^2)
    pred_accel = (pred_vel[:, 1:] - pred_vel[:, :-1]) / dt
    
    # 計算躍度 (m/s^3)
    pred_jerk = (pred_accel[:, 1:] - pred_accel[:, :-1]) / dt
    
    # 我們懲罰躍度向量的 L2 範數，而不是 L2 範數的平方，這樣更溫和
    # epsilon 確保在向量為零時也能穩定計算
    jerk_magnitude = torch.sqrt(torch.sum(pred_jerk**2, dim=-1) + epsilon)
    
    return torch.mean(jerk_magnitude)

# ==============================================================================
#  2. 【改進】魯棒的速度-轉向一致性損失 (Robust Velocity-Heading Loss)
# ==============================================================================
def velocity_heading_consistency_loss_v2(pred_seq, dt=0.05, epsilon=1e-8):
    """
    使用對數縮放來減弱高速對損失的極端影響，使其更魯棒。
    """
    if pred_seq.shape[1] < 3:
        return torch.tensor(0.0, device=pred_seq.device, dtype=pred_seq.dtype)

    # 1. 計算速度向量
    pred_vel = (pred_seq[:, 1:] - pred_seq[:, :-1]) / dt

    # 2. 計算速度大小 (speed)
    # 我們對速度大小進行裁剪，防止出現 log(0) 的情況
    pred_speed = torch.linalg.norm(pred_vel, dim=-1).clamp(min=epsilon)

    # 3. 計算航向變化
    pred_direction = F.normalize(pred_vel, p=2, dim=-1, eps=epsilon)
    # 點積的值等於 cos(theta)，將其裁剪到 [-1, 1] 區間以確保數值穩定性
    dot_products = (pred_direction[:, 1:] * pred_direction[:, :-1]).sum(dim=-1).clamp(-1.0, 1.0)
    heading_change_magnitude = 1.0 - dot_products

    # 4. 【核心改進】對速度進行對數縮放
    # 這使得損失的增長與速度的對數成正比，而不是與速度本身成正比。
    # 即使速度從 10m/s 增加到 100m/s，損失的增長也不會那麼劇烈。
    # 加 1 是為了確保 log 函數的輸入總是大於等於 1。
    log_scaled_speed = torch.log1p(pred_speed[:, :-1])

    # 5. 構造損失
    consistency_violation = log_scaled_speed * heading_change_magnitude
    
    return torch.mean(consistency_violation)