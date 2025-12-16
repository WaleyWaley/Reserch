import torch
import torch.nn.functional as F
import math
import torch.nn as nn
# import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd


class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,  # 网格大小，默认为 5
            spline_order=3,  # 分段多项式的阶数，默认为 3
            # 并根据 scale_noise 和 scale_spline 初始化 B-spline 权重。
            scale_noise=0.1,  # 缩放噪声，默认为 0.1
            scale_base=1.0,  # 基础缩放，默认为 1.0
            scale_spline=1.0,  # 分段多项式的缩放，默认为 1.0
            enable_standalone_scale_spline=True,
            base_activation=nn.SiLU,  # 基础激活函数，默认为 SiLU（Sigmoid Linear Unit）
            grid_eps=0.02,
            grid_range=[-1, 1],  # 网格范围，默认为 [-1, 1]
            dropout_rate=0.0
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features  # 输入特征数
        self.out_features = out_features  # 输出特征数
        self.grid_size = grid_size  # 设置网格大小和分段多项式的阶数
        self.spline_order = spline_order  # 样条阶数
        if dropout_rate>0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        # 网格步长（h）决定了网格点之间的距离，从而影响样条基函数的平滑程度和覆盖范围。
        h = (grid_range[1] - grid_range[0]) / grid_size  # 网格越密集，样条基函数的分辨率越高，可以更精细地拟合数据。

        # 生成网格
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features,
                                                                                                       -1).contiguous())

        self.register_buffer("grid", grid)  # 将网格作为缓冲区注册

        # 初始化基础权重参数，形状为 (out_features, in_features)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        # 初始化样条权重参数，形状为 (out_features, in_features, grid_size + spline_order)
        self.spline_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))

        # 如果启用了独立缩放样条功能，初始化样条缩放参数，形状为 (out_features, in_features)
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise  # 保存缩放噪声、基础缩放、分段多项式的缩放、是否启用独立的分段多项式缩放、基础激活函数和网格范围的容差
        # 基础权重的缩放系数，用于初始化基础权重时的缩放因子
        self.scale_base = scale_base
        # 样条权重的缩放系数，用于初始化样条权重时的缩放因子
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        # 基础激活函数实例，用于对输入进行非线性变换
        self.base_activation = base_activation()
        # 网格更新时的小偏移量，用于在更新网格时引入微小变化，避免过拟合
        self.grid_eps = grid_eps
        
        self.reset_parameters()  # 重置参数

        
    def reset_parameters(self):
        """替换 Kaiming 初始化 为 Xavier 初始化
        # Xavier 初始化用于激活函数为 Sigmoid 或 Tanh 的情况，适合于时序数据任务"""
        torch.nn.init.xavier_uniform_(self.base_weight, gain=math.sqrt(2) * self.scale_base)  # 使用 Xavier 均匀初始化基础权重
        # torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        with torch.no_grad():
            # 为样条权重参数spline_weight添加噪声进行初始化
            noise = ((torch.rand(self.grid_size + 1, self.in_features,
                                 self.out_features) - 1 / 2) * self.scale_noise / self.grid_size)
            # 计算样条权重参数的初始值，结合了scale_spline的缩放因子
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order], noise, ))
            if self.enable_standalone_scale_spline:  # 如果启用独立的分段多项式缩放，则使用 Kaiming 均匀初始化分段多项式缩放参数
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                # 作者此前使用了一般的初始化，效果不佳
                # torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
                torch.nn.init.xavier_uniform_(self.spline_scaler, gain=math.sqrt(2) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        计算给定输入张量的B样条基函数。
        B样条（B-splines）是一种用于函数逼近和插值的基函数。
        它们具有局部性、平滑性和数值稳定性等优点，广泛应用于计算机图形学、数据拟合和机器学习中。
        在这段代码中，B样条基函数用于在输入张量上进行非线性变换，以提高模型的表达能力。
        在KAN（Kolmogorov-Arnold Networks）模型中，B样条基函数用于将输入特征映射到高维空间中，以便在该空间中进行线性变换。
        具体来说，B样条基函数能够在给定的网格点上对输入数据进行插值和逼近，从而实现复杂的非线性变换。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
            torch.Tensor: B样条基函数张量，形状为 (batch_size, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        # 形状为 (in_features, grid_size + 2 * spline_order + 1)
        # 获取网格点（包含在buffer中的self.grid）
        grid: torch.Tensor = self.grid
        # 为了进行逐元素操作，将输入张量的最后一维扩展一维增加最后一维
        x = x.unsqueeze(-1)

        # 初始化B样条基函数的基矩阵
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        # 迭代计算样条基函数
        for k in range(1, self.spline_order + 1):
            bases = (((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1])
                     + ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]))
        # 确保B样条基函数的输出形状正确
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order,)
        # print(f'bases.shape: {bases.shape}')
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        计算插值给定点的曲线的系数。
        curve2coeff 方法用于计算插值给定点的曲线的系数。
        这些系数用于表示插值曲线在特定点的形状和位置。
        具体来说，该方法通过求解线性方程组来找到B样条基函数在给定点上的插值系数。
        此方法的作用是根据输入和输出点计算B样条基函数的系数，
        使得这些基函数能够精确插值给定的输入输出点对。
        这样可以用于拟合数据或在模型中应用非线性变换。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            y (torch.Tensor): 输出张量，形状为 (batch_size, in_features, out_features)。

        返回:
            torch.Tensor: 系数张量，形状为 (out_features, in_features, grid_size + spline_order)。
        """

        assert x.dim() == 2 and x.size(1) == self.in_features
        # 确保输出张量的形状正确
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        # 计算 B-样条基函数
        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        # print(f'A_shape: {A.shape}')
        
        # 转置输出张量
        B = y.transpose(0, 1)  # 形状为 (in_features, batch_size, out_features)
        # print(f'B_shape: {B.shape}')
        
        solution = torch.linalg.lstsq(  # 使用最小二乘法求解线性方程组
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)  # 形状为 (in_features, grid_size + spline_order, out_features)
        
        result = solution.permute(  # 调整结果的维度顺序
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        # 确保结果张量的形状正确
        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order,)
        # 返回连续存储的结果张量
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        """
        计算带有缩放因子的样条权重。

        样条缩放：如果启用了 enable_standalone_scale_spline，
        则将 spline_scaler 张量扩展一维后与 spline_weight 相乘，
        否则直接返回 spline_weight。

        具体来说，spline_weight 是一个三维张量，形状为 (out_features, in_features, grid_size + spline_order)。
        而 spline_scaler 是一个二维张量，形状为 (out_features, in_features)。
        为了使 spline_scaler 能够与 spline_weight 逐元素相乘，
        需要将 spline_scaler 的最后一维扩展，以匹配 spline_weight 的第三维。

        返回:
            torch.Tensor: 带有缩放因子的样条权重张量。
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):  # 将输入数据通过模型的各个层，经过线性变换和激活函数处理，最终得到模型的输出结果
        """
        前向传播函数。
        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
        返回:
        torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        # 确保输入张量的最后一维大小等于输入特征数
        original_shape = x.shape
        # assert x.dim() == 2 and x.size(1) == self.in_features
        if x.dim() == 3:
            # x = x.view(-1, self.in_features)  # 展平时间维度
            x = x.reshape(-1,self.in_features)
        """
            # 保存输入张量的原始形状
            original_shape = x.shape
        
            # 将输入张量展平为二维
            x = x.view(-1, self.in_features)
        """

        # 计算基础线性变换的输出
        base_output = F.linear(self.base_activation(x), self.base_weight)

        if self.dropout is not None:
            base_output=self.dropout(base_output)
            
        # 计算B样条基函数的输出
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1),
                                 self.scaled_spline_weight.view(self.out_features, -1), )
        """
            # 合并基础输出和样条输出
            output = base_output + spline_output
            # 恢复输出张量的形状
            output = output.view(*original_shape[:-1], self.out_features)
        """
        output = base_output + spline_output
        # 恢复时序形状：(batch*time, out) -> (batch, time, out)
        if len(original_shape) == 3:
            output = output.view(original_shape[0], original_shape[1], self.out_features)
        return output

    @torch.no_grad()
    # 更新网格。
    # 参数:
    # x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
    # margin (float): 网格边缘空白的大小。默认为 0.01。
    # 根据输入数据 x 的分布情况来动态更新模型的网格,使得模型能够更好地适应输入数据的分布特点，从而提高模型的表达能力和泛化能力。
    def update_grid(self, x: torch.Tensor, margin=0.01):
        """
        update_grid 方法用于根据输入数据动态更新B样条的网格点，从而适应输入数据的分布。
        该方法通过重新计算和调整网格点，确保B样条基函数能够更好地拟合数据。
        这在训练过程中可能会提高模型的精度和稳定性。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            margin (float): 网格更新的边缘大小，用于在更新网格时引入微小变化。
        """
        # 确保输入张量的维度正确
        # assert x.dim() == 2 and x.size(1) == self.in_features
        # batch = x.size(0)   # 获取批量大小
        # 处理时序输入：(batch, time, features) -> (batch*time, features)
        if x.dim() == 3:
            x = x.reshape(-1, self.in_features)
        # 计算输入张量的B样条基函数
        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # 转置为 (in, batch, coeff)

        # 获取当前的样条权重
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # 转置为 (in, coeff, out)

        # 计算未缩减的样条输出
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)  # 转置为 (batch, in, out)

        # 为了收集数据分布，对每个通道分别进行排序
        x_sorted = torch.sort(x, dim=0)[0]
        batch = x_sorted.size(0)
        grid_adaptive = x_sorted[torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)]

        # 计算均匀步长，并生成均匀网格
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        # 混合均匀网格和自适应网格
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        # 扩展网格以包括样条边界
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        # 更新模型中的网格点
        self.grid.copy_(grid.T)

        # 重新计算样条权重
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # 计算正则化损失，用于约束模型的参数，防止过拟合
        """
        计算正则化损失。

        这是对原始 L1 正则化的简单模拟，因为原始方法需要从扩展的（batch, in_features, out_features）中间张量计算绝对值和熵，
        而这个中间张量被 F.linear 函数隐藏起来，如果我们想要一个内存高效的实现。

        现在的 L1 正则化是计算分段多项式权重的平均绝对值。作者的实现也包括这一项，除了基于样本的正则化。

        参数:
        regularize_activation (float): 正则化激活项的权重，默认为 1.0。
        regularize_entropy (float): 正则化熵项的权重，默认为 1.0。

        返回:
        torch.Tensor: 正则化损失。
        """
        eps = 1e-8  # 防止 log(0)
        # 计算样条权重的绝对值的平均值
        l1_fake = self.spline_weight.abs().mean(-1)
        # 计算激活正则化损失，即所有样条权重绝对值的和
        regularization_loss_activation = l1_fake.sum()
        # 计算每个权重占总和的比例
        p = l1_fake / regularization_loss_activation
        # 计算熵正则化损失，即上述比例的负熵
        # regularization_loss_entropy = -torch.sum(p * p.log())
        regularization_loss_entropy = -torch.sum(p * torch.log(p + eps))
        # 返回总的正则化损失，包含激活正则化和熵正则化
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):  # 封装了一个KAN神经网络模型，可以用于对数据进行拟合和预测。
    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            use_lstm=True,
            lstm_hidden=64,
            dropout_rate=0.0
    ):
        """
        初始化 KAN 模型。
        参数:
            layers_hidden (list): 每层的输入和输出特征数列表。
            grid_size (int): 网格大小，默认为 5。
            spline_order (int): 样条阶数，默认为 3。
            scale_noise (float): 样条权重初始化时的噪声缩放系数，默认为 0.1。
            scale_base (float): 基础权重初始化时的缩放系数，默认为 1.0。
            scale_spline (float): 样条权重初始化时的缩放系数，默认为 1.0。
            base_activation (torch.nn.Module): 基础激活函数类，默认为 SiLU。
            grid_eps (float): 网格更新时的小偏移量，默认为 0.02。
            grid_range (list): 网格范围，默认为 [-1, 1]。
        """
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.use_lstm = use_lstm
        self.dropout_rate = dropout_rate
        # 初始化模型层
        self.layer_norm_lstm = nn.LayerNorm(lstm_hidden) # 或 lstm_hidden*2 如果是双向
        self.layers = torch.nn.ModuleList()
        # 时序层（LSTM）
        if use_lstm:
            self.lstm = torch.nn.LSTM(
                input_size=layers_hidden[0],
                hidden_size=lstm_hidden,
                num_layers=3,
                batch_first=True,
                dropout=0.0,
            )
            if dropout_rate>0:
                self.lstm_dropout = nn.Dropout(dropout_rate)
            else:
                self.lstm_dropout = None
            layers_hidden[0] = lstm_hidden  # 更新输入维度
            
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    dropout_rate=dropout_rate # 传递dropout率
                )
            )
            self.layers.append(nn.LayerNorm(out_features))
            
        # 使用nn.Parameter将它们注册为模型的参数
        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.beta = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.gamma = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        # 可以添加softplus激活确保参数为正
        self.activation = nn.Softplus()
        
        
    def forward(self, x: torch.Tensor, update_grid=False):  # 调用每个KANLinear层的forward方法，对输入数据进行前向传播计算输出。
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            update_grid (bool): 是否更新网格。默认为 False。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        # 处理时序输入：(batch, time, features)
        self.input_for_temporal = x  # 保存输入用于时序正则化

        if self.use_lstm:
            x, _ = self.lstm(x)  # (batch, time, lstm_hidden)
            x = self.layer_norm_lstm(x)
            # LSTM后应用dropout
            if self.lstm_dropout is not None:
                x = self.lstm_dropout(x)
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x.reshape(-1, x.size(-1)))  # 展平时间维度更新网格
            x_input = x
            x = layer(x)  # 保持时序维度：(batch, time, out_features)
            if(x_input.shape == x.shape):   # 残差
                x = x + x_input
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """修改后"""
        # 遍歷 ModuleList 中的每一個 KANLinear 層
        # ### 關鍵修正：在使用前，先將累加器初始化為 0.0 ###
        total_reg_loss = 0.0
        for layer in self.layers:
            # 呼叫【每一層自己】的正則化損失方法，並累加
            total_reg_loss += layer.regularization_loss(
                regularize_activation, regularize_entropy
            )
        
        return total_reg_loss


    def get_loss_weights(self):
        # 返回经过激活后的权重，确保为正值
        return self.activation(self.alpha), self.activation(self.beta), self.activation(self.gamma)
        
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="训练损失", color='blue', linestyle='-', marker='o')
    plt.plot(val_losses, label="验证损失", color='green', linestyle='-', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失与验证损失')
    plt.legend()
    plt.grid(True)
    plt.show()
