import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
#  第一部分：KANLinear 模組 (来自您最初成功但有数值风险的版本)
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
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

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
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order], noise
                )
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
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        if self.dropout is not None:
            base_output = self.dropout(base_output)
            
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        if len(original_shape) == 3:
            output = output.view(original_shape[0], original_shape[1], self.out_features)
        return output
        
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        eps = 1e-8
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / (regularization_loss_activation + eps)
        regularization_loss_entropy = -torch.sum(p * torch.log(p + eps))
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

# ==============================================================================
#  第二部分：KAN 主类 (采用您最初成功的“增量学习”架构)
# ==============================================================================
class KAN(nn.Module):
    """
    此版本 KAN 采用“增量学习”架构，已被证明在您的任务上有效。
    它内部使用未经修改的原始 KANLinear 模組。
    """
    def __init__(self, layers_hidden, use_lstm=True, lstm_hidden=64, **kan_kwargs):
        super(KAN, self).__init__()
        self.use_lstm = use_lstm

        # 1. 可选的 LSTM 前置层
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=layers_hidden[0],
                hidden_size=lstm_hidden,
                num_layers=1,
                batch_first=True,
                dropout=kan_kwargs.get('dropout_rate', 0.0),
            )
            kan_input_dim = lstm_hidden
            self.layer_norm_lstm = nn.LayerNorm(lstm_hidden)
        else:
            self.lstm = None
            kan_input_dim = layers_hidden[0]

        # 2. 构建旧版的交错层结构 (KANLinear -> LayerNorm -> KANLinear -> ...)
        self.kan_layers = nn.ModuleList()
        kan_layers_dims = [kan_input_dim] + layers_hidden[1:]
        
        for i, (in_features, out_features) in enumerate(zip(kan_layers_dims, kan_layers_dims[1:])):
            self.kan_layers.append(
                KANLinear(in_features, out_features, **kan_kwargs)
            )
            # 在非最后一层 KANLinear 之后添加 LayerNorm
            if i < len(kan_layers_dims) - 2:
                self.kan_layers.append(nn.LayerNorm(out_features))

    def forward(self, x: torch.Tensor):
        if self.use_lstm:
            x, _ = self.lstm(x)
            x = self.layer_norm_lstm(x)

        # 采用旧版的“增量修正”前向传播逻辑
        for layer in self.kan_layers:
            x_input = x
            x = layer(x)
            if x_input.shape == x.shape:
                x = x + x_input
        
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # 只对 KANLinear 层计算正则化损失
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.kan_layers if isinstance(layer, KANLinear)
        )