import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
#  第一部分：KANLinear 模块 (源自您性能最佳的 LSTKAN_Ori.py)
#  我们保留了这个核心的计算单元，因为它已被证明具有强大的拟合能力。
# ==============================================================================
class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            dropout_rate=0.0
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 为 KANLinear 层自身加入 Dropout
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
                .expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.base_weight, gain=math.sqrt(2) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5) *
                     self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0) *
                self.curve2coeff(self.grid.T[self.spline_order: -self.spline_order], noise)
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.xavier_uniform_(self.spline_scaler, gain=math.sqrt(2) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                ((x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)])) * bases[:, :, :-1]
            ) + (
                ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)])) * bases[:, :, 1:]
            )
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        
        # 添加微小的扰动以增强数值稳定性，防止秩亏问题
        A = A + 1e-6 * torch.randn_like(A)
        
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0
        )

    def _prepare_input(self, x: torch.Tensor):
        """私有辅助函数，用于处理2D或3D输入，实现代码复用。"""
        original_shape = x.shape
        is_3d = x.dim() == 3
        if is_3d:
            x = x.reshape(-1, self.in_features)
        return x, original_shape, is_3d

    def forward(self, x: torch.Tensor):
        x, original_shape, is_3d = self._prepare_input(x)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1)
        )
        
        output = base_output + spline_output

        # 在 KANLinear 内部应用 Dropout
        if self.dropout is not None:
            output = self.dropout(output)

        if is_3d:
            output = output.view(original_shape[0], original_shape[1], self.out_features)
            
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        x, _, _ = self._prepare_input(x)
        
        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, x_sorted.size(0) - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]
        
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] - margin
        )
        
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        
        grid = torch.cat([
            grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
            grid,
            grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
        ], dim=0)
        
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        eps = 1e-8
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / (regularization_loss_activation + eps)
        regularization_loss_entropy = -torch.sum(p * torch.log(p + eps))
        return (
            regularize_activation * regularization_loss_activation +
            regularize_entropy * regularization_loss_entropy
        )

# ==============================================================================
#  第二部分：改进版 KAN 模块 (融合了 LSTKAN.py 的 ResNet 稳健架构)
#  这是我们抑制过拟合、提升泛化能力的关键。
# ==============================================================================
class KAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            use_lstm=True,
            lstm_hidden=64,
            dropout_rate=0.0,
            num_layers = 1,
            **kan_kwargs  # 接收所有 KANLinear 的参数, 如 grid_size, spline_order
    ):
        super(KAN, self).__init__()
        self.use_lstm = use_lstm

        # 1. 定义可选的 LSTM 层，并为其配备 Dropout 和 LayerNorm
        if use_lstm:
            self.lstm = torch.nn.LSTM(
                input_size=layers_hidden[0],
                hidden_size=lstm_hidden,
                num_layers=num_layers, # 增加 LSTM 层数以提升表达能力
                batch_first=True,
                dropout=dropout_rate, # 在 LSTM 层之间应用 Dropout
                # bidirectional=True  # <--- 开启双向
            )
            self.layer_norm_lstm = nn.LayerNorm(lstm_hidden)
            kan_input_dim = lstm_hidden
            # 双向 LSTM 的输出维度是 hidden * 2
            # self.layer_norm_lstm = nn.LayerNorm(lstm_hidden * 2) 
            # kan_input_dim = lstm_hidden * 2 # <--- 输入给 KAN 的维度变大
        else:
            self.lstm = None
            kan_input_dim = layers_hidden[0]

        # 2. 【核心改进】构建 ResNet 风格的 KAN 块
        self.kan_layers = torch.nn.ModuleList()
        self.res_projs = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()

        kan_layers_dims = [kan_input_dim] + layers_hidden[1:]
        for in_features, out_features in zip(kan_layers_dims, kan_layers_dims[1:]):
            # a. KAN 变换层
            self.kan_layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    dropout_rate=dropout_rate, # 将 Dropout 传递给 KANLinear
                    **kan_kwargs
                )
            )
            # b. 用于残差连接的投影层（处理维度变化）
            if in_features != out_features:
                self.res_projs.append(nn.Linear(in_features, out_features))
            else:
                self.res_projs.append(nn.Identity())
            
            # c. 归一化层
            self.norm_layers.append(nn.LayerNorm(out_features))

    def forward(self, x: torch.Tensor, update_grid=False):
        # 1. 可选的 LSTM 前处理
        if self.use_lstm:
            x, _ = self.lstm(x)
            x = self.layer_norm_lstm(x) # LSTM 输出后进行归一化

        # 2. 逐个通过 ResNet 风格的 KAN 块
        for i in range(len(self.kan_layers)):
            residual = self.res_projs[i](x)
            
            if update_grid:
                self.kan_layers[i].update_grid(x)

            x_kan = self.kan_layers[i](x)

            # 【核心架构】先相加，再归一化，最后激活
            x = self.norm_layers[i](x_kan + residual)
            x = F.silu(x)

        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """采用更 Pythonic 的写法计算总正则化损失"""
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.kan_layers
        )