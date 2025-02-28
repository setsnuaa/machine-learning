import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot


# ====================== 1. 修复后的手动图注意力层（GAT） ======================
class ManualGATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.out_features = out_features

        # 线性变换参数（共享权重）
        self.W = nn.Linear(in_features, out_features * heads, bias=False)
        # 注意力参数（每个头独立）
        self.a_src = nn.Parameter(torch.Tensor(1, heads, out_features))  # 修正形状
        self.a_dst = nn.Parameter(torch.Tensor(1, heads, out_features))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重参数"""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, edge_index):
        N = x.size(0)
        E = edge_index.size(1)
        h = self.heads

        # 1. 线性变换 + 分头
        x_trans = self.W(x).view(N, h, self.out_features)  # [N, heads, out_features]

        # 2. 提取源节点和目标节点特征
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        x_src = x_trans[src_nodes]  # [E, heads, out_features]
        x_dst = x_trans[dst_nodes]  # [E, heads, out_features]

        # 3. 计算注意力分数（修正点：使用矩阵乘法替代逐元素乘）
        # 计算 a_src^T * x_src 和 a_dst^T * x_dst
        attn_src = (x_src * self.a_src).sum(dim=-1)  # [E, heads]
        attn_dst = (x_dst * self.a_dst).sum(dim=-1)
        attn_scores = attn_src + attn_dst  # [E, heads]
        attn_scores = self.leaky_relu(attn_scores)

        # 4. 归一化注意力权重
        attn_weights = F.softmax(attn_scores, dim=0)  # [E, heads]
        attn_weights = self.dropout(attn_weights)

        # 5. 特征聚合（加权求和）
        out = torch.zeros(N, h, self.out_features, device=x.device)
        for head in range(h):
            # 对每个头单独聚合
            weighted_features = x_src[:, head, :] * attn_weights[:, head].unsqueeze(-1)  # [E, out_features]
            out[:, head, :].scatter_add_(
                dim=0,
                index=dst_nodes.unsqueeze(-1).expand(-1, self.out_features),
                src=weighted_features
            )

        return out.view(N, h * self.out_features)  # [N, heads*out_features]


# ====================== 2. 时间卷积网络（TCN） ======================
class TCN(nn.Module):
    def __init__(self, input_size=24, hidden_size=64, kernel_size=3, dilation=2):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2  # 调整 padding 保持输出长度一致
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [Batch, Time, Nodes] → [Batch, Nodes, Time]
        x = self.conv(x)  # [Batch, Hidden, Time]
        x = x.permute(0, 2, 1)  # [Batch, Time, Hidden]
        return self.activation(x)


# ====================== 3. 时空融合模块（保持不变） ======================
class FusionModule(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.W_g = nn.Linear(2 * input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_space, h_time):
        combined = torch.cat([h_space, h_time], dim=-1)
        g = self.sigmoid(self.W_g(combined))
        return g * h_space + (1 - g) * h_time


# ====================== 4. 整体模型（ST-GNN） ======================
class STGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat = ManualGATLayer(in_features=70, out_features=16, heads=4, dropout=0.2)
        self.tcn = TCN(input_size=24, hidden_size=64)
        self.fusion = FusionModule(input_dim=64)
        self.output = nn.Linear(64, 1)

    def forward(self, x_spatial, edge_index, x_temporal):
        h_space = self.gat(x_spatial, edge_index)  # [N, 64]
        h_time = self.tcn(x_temporal).squeeze(0)  # [N, 64]
        h_fused = self.fusion(h_space, h_time)
        return torch.sigmoid(self.output(h_fused))


# ====================== 5. 示例与验证 ======================
if __name__ == "__main__":
    # 生成模拟数据
    N = 100  # 节点数
    E = 200  # 边数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_spatial = torch.randn(N, 70).to(device)  # 空间特征 [N, 70]
    edge_index = torch.randint(0, N, (2, E)).to(device)  # 边索引 [2, E]
    x_temporal = torch.randn(1, N, 24).to(device)  # 时间特征 [Batch=1, N, 24]

    # 初始化模型
    model = STGNN().to(device)

    # 前向传播
    output = model(x_spatial, edge_index, x_temporal)
    print("输出形状:", output.shape)  # 应输出 [N, 1]

    # 可视化计算图
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render("STGNN_Architecture", format="png", cleanup=True)
    print("网络结构图已保存为 STGNN_Architecture.png")