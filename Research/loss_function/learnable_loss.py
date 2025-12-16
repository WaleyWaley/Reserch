import torch
import torch.nn as nn

class LearnableLossWeighter(nn.Module):
    """
    一個 nn.Module，用於管理多個損失項的可學習權重。
    基於 "Multi-Task Learning Using Uncertainty to Weigh Losses" 的思想。
    """
    def __init__(self, num_losses):
        """
        初始化函數。
        參數:
            num_losses (int): 需要平衡的損失項的數量。
        """
        super(LearnableLossWeighter, self).__init__()
        self.num_losses = num_losses
        
        # 我們學習的是 log(sigma^2)，這是一個無約束的變量，更易於優化。
        # 初始化為 0，意味著初始時 sigma^2 = 1，每個損失的初始權重約為 1。
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, *losses):
        """
        計算加權後的總損失。
        參數:
            *losses: 一系列標量的損失張量，其數量必須與 num_losses 匹配。
        """
        if len(losses) != self.num_losses:
            raise ValueError(f"預期有 {self.num_losses} 個損失項，但收到了 {len(losses)} 個。")

        total_loss = 0
        for i, loss in enumerate(losses):
            # 應用公式: exp(-log_var) * loss + log_var
            # 這在數學上等效於 1/(sigma^2) * loss + 2*log(sigma)，並且數值上更穩定
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            total_loss += weighted_loss
            
        return total_loss
