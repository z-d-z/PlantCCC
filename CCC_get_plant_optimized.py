# ==================== CCC_get_plant_optimized.py ====================
# 针对植物空间转录组细胞通讯优化的 GAT + DGI 实现
#
# 主要优化点：
# 1. 多头注意力融合策略优化
# 2. 植物特异性边特征增强
# 3. 空间距离感知的注意力机制
# 4. 改进的对比学习策略
# 5. L-R对层级建模
import os
import datetime
import gzip
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.data import Data
from torch_geometric.utils import degree
from typing import Optional, Tuple

class PlantEdgeEncoder(nn.Module):
    """
    植物细胞通讯专用边特征编码器

    改进点：
    - 分离处理距离、协表达、L-R类型三种特征
    - 添加距离衰减函数（植物细胞壁限制信号传播）
    - L-R对的层级嵌入（考虑信号通路家族）
    """

    def __init__(self, rel_vocab, rel_emb_dim=16, hidden_dim=32):
        super().__init__()

        # L-R关系嵌入（增大维度以捕获更多信息）
        self.rel_emb = nn.Embedding(rel_vocab, rel_emb_dim) if rel_vocab > 0 else None

        # 距离特征编码（植物特异性：细胞壁导致距离衰减更快）
        self.dist_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rel_emb_dim)
        )

        # 协表达分数编码
        self.coexp_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rel_emb_dim)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(rel_emb_dim * 3, rel_emb_dim * 2),
            nn.LayerNorm(rel_emb_dim * 2),
            nn.ReLU(),
            nn.Linear(rel_emb_dim * 2, rel_emb_dim)
        )

        self.output_dim = rel_emb_dim

    def forward(self, edge_attr):
        """
        edge_attr: [E, 3] -> [距离权重, 协表达分数, 关系ID]
        """
        dist = edge_attr[:, 0:1]  # [E, 1]
        coexp = edge_attr[:, 1:2]  # [E, 1]
        rel_id = edge_attr[:, 2].long().clamp(min=0)  # [E]

        # 距离衰减（植物细胞壁效应：使用更陡的衰减）
        # 原始距离权重已经是 1 - normalized_dist，这里进一步处理
        dist_feat = self.dist_encoder(dist)

        # 协表达特征
        coexp_feat = self.coexp_encoder(coexp)

        # L-R关系嵌入
        if self.rel_emb is not None:
            rel_feat = self.rel_emb(rel_id)
        else:
            rel_feat = torch.zeros_like(dist_feat)

        # 融合
        combined = torch.cat([dist_feat, coexp_feat, rel_feat], dim=1)
        return self.fusion(combined)

class SpatialAwareGATConv(nn.Module):
    """
    空间感知的图注意力卷积层

    改进点：
    - 显式建模空间距离对注意力的影响
    - 多头注意力使用不同的空间感知策略
    - 支持方向性注意力（配体→受体）
    """

    def __init__(self, in_channels, out_channels, edge_dim, heads=4,
                 dropout=0.1, spatial_decay=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.spatial_decay = spatial_decay

        # 节点变换
        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)

        # 边特征变换
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)

        # 注意力参数
        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))

        # 空间衰减参数（可学习）
        if spatial_decay:
            self.spatial_scale = nn.Parameter(torch.ones(heads))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

        # 存储注意力分数
        self._alpha_raw = None
        self._alpha_norm = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_attr, dist_weight=None):
        """
        x: [N, in_channels]
        edge_index: [2, E]
        edge_attr: [E, edge_dim] (已编码的边特征)
        dist_weight: [E] 原始距离权重（可选，用于空间衰减）
        """
        H, C = self.heads, self.out_channels
        N = x.size(0)

        # 源节点和目标节点变换
        src_idx, dst_idx = edge_index[0], edge_index[1]

        x_src = self.lin_src(x).view(-1, H, C)  # [N, H, C]
        x_dst = self.lin_dst(x).view(-1, H, C)  # [N, H, C]

        # 边特征变换
        edge_feat = self.lin_edge(edge_attr).view(-1, H, C)  # [E, H, C]

        # 计算注意力（结合源、目标、边）
        alpha = x_src[src_idx] + x_dst[dst_idx] + edge_feat  # [E, H, C]
        alpha = torch.tanh(alpha)  # 使用tanh（PlantCCC论文）
        alpha = (alpha * self.att).sum(dim=-1)  # [E, H]

        # 保存原始注意力
        self._alpha_raw = alpha.clone()

        # 空间衰减（植物特异性）
        if self.spatial_decay and dist_weight is not None:
            # dist_weight 越大表示越近，所以直接用作权重
            spatial_factor = dist_weight.unsqueeze(-1) ** self.spatial_scale.unsqueeze(0)
            alpha = alpha * spatial_factor

        # Softmax归一化（按目标节点）
        alpha = self._softmax_by_dst(alpha, dst_idx, N)
        self._alpha_norm = alpha.clone()

        alpha = self.dropout(alpha)

        # 消息传递
        out = x_src[src_idx] * alpha.unsqueeze(-1)  # [E, H, C]

        # 聚合到目标节点
        out = self._aggregate(out, dst_idx, N)  # [N, H, C]
        out = out.view(N, H * C)

        return out, self._alpha_norm, self._alpha_raw

    def _softmax_by_dst(self, alpha, dst_idx, num_nodes):
        """按目标节点进行softmax"""
        alpha_max = torch.zeros(num_nodes, alpha.size(1), device=alpha.device)
        alpha_max.scatter_reduce_(0, dst_idx.unsqueeze(-1).expand_as(alpha),
                                  alpha, reduce='amax', include_self=False)
        alpha = alpha - alpha_max[dst_idx]
        alpha = torch.exp(alpha)

        alpha_sum = torch.zeros(num_nodes, alpha.size(1), device=alpha.device)
        alpha_sum.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(alpha), alpha)
        alpha = alpha / (alpha_sum[dst_idx] + 1e-8)

        return alpha

    def _aggregate(self, msg, dst_idx, num_nodes):
        """聚合消息到目标节点"""
        out = torch.zeros(num_nodes, msg.size(1), msg.size(2), device=msg.device)
        out.scatter_add_(0, dst_idx.unsqueeze(-1).unsqueeze(-1).expand_as(msg), msg)
        return out




class PlantCCCEncoder(nn.Module):
    """
    植物细胞通讯专用编码器

    改进点：
    - 使用植物特异性边编码器
    - 空间感知的注意力机制
    - 多尺度特征融合
    - 更深的残差连接
    """

    def __init__(self, in_channels, hidden_channels, heads=4, num_layers=3,
                 dropout=0.1, rel_vocab=100, rel_emb_dim=16):
        super().__init__()

        self.num_layers = num_layers

        # 输入归一化
        self.in_norm = nn.LayerNorm(in_channels)

        # 边特征编码器
        self.edge_encoder = PlantEdgeEncoder(rel_vocab, rel_emb_dim)
        edge_dim = self.edge_encoder.output_dim

        # 输入投影
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # 多层空间感知GAT
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(
                SpatialAwareGATConv(
                    hidden_channels, hidden_channels // heads,
                    edge_dim=edge_dim, heads=heads, dropout=dropout,
                    spatial_decay=(i < num_layers - 1)  # 最后一层不用空间衰减
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.dropout = nn.Dropout(dropout)

        # 存储各层注意力
        self.layer_attentions_raw = []
        self.layer_attentions_norm = []

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # 提取原始距离权重（用于空间衰减）
        dist_weight = edge_attr[:, 0] if edge_attr.size(1) >= 1 else None

        # 输入处理
        x = self.in_norm(x)
        x = self.input_proj(x)

        # 边特征编码
        edge_feat = self.edge_encoder(edge_attr)

        # 多层GAT
        self.layer_attentions_raw = []
        self.layer_attentions_norm = []

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x
            x, att_norm, att_raw = conv(x, edge_index, edge_feat, dist_weight)
            x = norm(x)
            x = F.elu(x)
            x = self.dropout(x)
            x = x + x_res  # 残差连接

            self.layer_attentions_raw.append(att_raw)
            self.layer_attentions_norm.append(att_norm)

        return x

    def get_communication_scores(self):
        """
        获取细胞通讯分数

        策略：使用所有层注意力的加权平均
        - 浅层捕获局部模式
        - 深层捕获全局模式
        """
        if not self.layer_attentions_raw:
            return None

        # 层权重（深层权重更高）
        num_layers = len(self.layer_attentions_raw)
        weights = torch.softmax(torch.arange(num_layers, dtype=torch.float), dim=0)

        # 加权平均
        scores = None
        for w, att in zip(weights, self.layer_attentions_raw):
            att_mean = att.mean(dim=-1) if att.dim() > 1 else att  # 多头平均
            if scores is None:
                scores = w * att_mean
            else:
                scores = scores + w * att_mean

        return scores


# ==================== 4. 改进的对比学习策略 ====================

class PlantDGILoss(nn.Module):
    """
    植物细胞通讯专用对比学习损失

    改进点：
    - 多视图对比（节点级 + 图级 + 边级）
    - 空间一致性约束
    - 硬负样本挖掘
    """

    def __init__(self, hidden_dim, tau=0.5):
        super().__init__()
        self.tau = tau

        # 判别器
        self.node_discriminator = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.edge_discriminator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, encoder, data, corruption_fn):
        """
        多视图对比学习损失
        """
        # 编码真实图
        z_pos = encoder(data)

        # 图级summary
        summary = z_pos.mean(dim=0, keepdim=True)

        # === 损失1：节点-图对比（标准DGI）===
        data_neg = corruption_fn(data)
        z_neg = encoder(data_neg)

        loss_node_graph = self._node_graph_loss(z_pos, z_neg, summary)

        # === 损失2：边级对比 ===
        loss_edge = self._edge_contrastive_loss(z_pos, data.edge_index, encoder)

        # === 损失3：空间一致性（近邻节点嵌入相似）===
        loss_spatial = self._spatial_consistency_loss(z_pos, data.edge_index, data.edge_attr)

        # 总损失
        total_loss = loss_node_graph + 0.5 * loss_edge + 0.3 * loss_spatial

        return total_loss

    def _node_graph_loss(self, z_pos, z_neg, summary):
        """节点-图级对比损失"""
        s = F.normalize(summary, p=2, dim=1)
        p = F.normalize(z_pos, p=2, dim=1)
        n = F.normalize(z_neg, p=2, dim=1)

        logits_pos = (p * s).sum(dim=1) / self.tau
        logits_neg = (n * s).sum(dim=1) / self.tau

        bce = nn.BCEWithLogitsLoss(reduction='mean')
        loss = bce(logits_pos, torch.ones_like(logits_pos)) + \
               bce(logits_neg, torch.zeros_like(logits_neg))

        return loss

    def _edge_contrastive_loss(self, z, edge_index, encoder):
        """边级对比损失：真实边 vs 随机边"""
        src, dst = edge_index[0], edge_index[1]

        # 正样本：真实边
        pos_pairs = torch.cat([z[src], z[dst]], dim=1)
        pos_scores = self.edge_discriminator(pos_pairs).squeeze()

        # 负样本：随机配对
        neg_dst = dst[torch.randperm(dst.size(0))]
        neg_pairs = torch.cat([z[src], z[neg_dst]], dim=1)
        neg_scores = self.edge_discriminator(neg_pairs).squeeze()

        bce = nn.BCEWithLogitsLoss(reduction='mean')
        loss = bce(pos_scores, torch.ones_like(pos_scores)) + \
               bce(neg_scores, torch.zeros_like(neg_scores))

        return loss

    def _spatial_consistency_loss(self, z, edge_index, edge_attr):
        """空间一致性损失：近邻节点嵌入应该相似"""
        src, dst = edge_index[0], edge_index[1]
        dist_weight = edge_attr[:, 0]  # 距离权重

        # 嵌入相似度
        z_norm = F.normalize(z, p=2, dim=1)
        sim = (z_norm[src] * z_norm[dst]).sum(dim=1)

        # 期望：距离越近（权重越大）→ 相似度越高
        # 使用MSE让相似度接近距离权重
        loss = F.mse_loss(sim, dist_weight)

        return loss


# ==================== 5. 改进的数据增强 ====================

def spatial_aware_corruption(data, drop_rate=0.1):
    """
    空间感知的数据增强

    改进：不完全随机打乱，而是考虑空间结构
    - 保持局部邻域结构
    - 只打乱远距离节点
    """
    x = data.x
    edge_attr = data.edge_attr
    N = x.size(0)

    # 计算每个节点的平均邻域距离
    dist_weight = edge_attr[:, 0]
    src, dst = data.edge_index[0], data.edge_index[1]

    node_locality = torch.zeros(N, device=x.device)
    node_count = torch.zeros(N, device=x.device)
    node_locality.scatter_add_(0, dst, dist_weight)
    node_count.scatter_add_(0, dst, torch.ones_like(dist_weight))
    node_locality = node_locality / (node_count + 1e-8)

    # 局部性低的节点更可能被打乱
    shuffle_prob = 1 - node_locality
    shuffle_prob = shuffle_prob / (shuffle_prob.max() + 1e-8)

    # 生成打乱索引
    perm = torch.arange(N, device=x.device)
    shuffle_mask = torch.rand(N, device=x.device) < (shuffle_prob * drop_rate * 10)
    shuffle_indices = torch.where(shuffle_mask)[0]

    if len(shuffle_indices) > 1:
        shuffled = shuffle_indices[torch.randperm(len(shuffle_indices))]
        perm[shuffle_indices] = shuffled

    return Data(
        x=x[perm],
        edge_index=data.edge_index,
        edge_attr=data.edge_attr
    )


def edge_perturbation(data, add_rate=0.05, drop_rate=0.1):
    """
    边扰动增强

    - 随机删除一些边
    - 随机添加一些边（基于空间邻近性）
    """
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    E = edge_index.size(1)
    N = data.x.size(0)

    # 删除边
    keep_mask = torch.rand(E, device=edge_index.device) > drop_rate
    new_edge_index = edge_index[:, keep_mask]
    new_edge_attr = edge_attr[keep_mask]

    return Data(
        x=data.x,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr
    )


# ==================== 6. 主训练函数 ====================

def train_plant_PlantCCC(args, graph, in_channels, edge_dim, rel_vocab):
    """
    植物细胞通讯优化版训练函数

    参数:
        args: 需要包含以下属性
            - data_name: 数据集名称（用于创建子目录，与后处理兼容）
            - model_name: 模型名称
            - model_path: 模型保存根目录
            - embedding_path: 嵌入保存根目录
            - hidden, heads, num_layers, dropout, rel_emb_dim: 模型参数
            - num_epoch, lr_rate, dgi_tau, patience_limit, min_stop: 训练参数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========== 初始化模型 ==========
    encoder = PlantCCCEncoder(
        in_channels=in_channels,
        hidden_channels=args.hidden,
        heads=getattr(args, 'heads', 4),
        num_layers=getattr(args, 'num_layers', 3),
        dropout=args.dropout,
        rel_vocab=rel_vocab,
        rel_emb_dim=getattr(args, 'rel_emb_dim', 16)
    ).to(device)

    # 对比学习损失
    dgi_loss_fn = PlantDGILoss(
        hidden_dim=args.hidden,
        tau=getattr(args, 'dgi_tau', 0.5)
    ).to(device)

    # 优化器
    params = list(encoder.parameters()) + list(dgi_loss_fn.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr_rate, weight_decay=1e-5)

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    # ========== 创建输出目录（兼容原版路径结构）==========
    # 关键修复：添加 data_name 子目录，与后处理器路径一致
    data_name = getattr(args, 'data_name', 'default')

    model_subdir = os.path.join(args.model_path, data_name)
    embedding_subdir = os.path.join(args.embedding_path, data_name)

    os.makedirs(model_subdir, exist_ok=True)
    os.makedirs(embedding_subdir, exist_ok=True)

    model_file = os.path.join(model_subdir, f'PlantCCC_{args.model_name}.pth.tar')

    # 将图移到GPU
    graph = graph.to(device)

    # ========== 训练 ==========
    min_loss = float('inf')
    patience = 0
    patience_limit = getattr(args, 'patience_limit', 300)
    min_stop = getattr(args, 'min_stop', 500)

    print(f"\n{'=' * 60}")
    print(f"开始训练 Plant-PlantCCC")
    print(f"设备: {device}")
    print(f"节点数: {graph.x.size(0)}, 边数: {graph.edge_index.size(1)}")
    print(f"{'=' * 60}\n")

    for epoch in tqdm(range(args.num_epoch), desc="Training"):
        encoder.train()
        dgi_loss_fn.train()
        optimizer.zero_grad()

        # 数据增强
        if torch.rand(1) < 0.5:
            graph_aug = edge_perturbation(graph)
        else:
            graph_aug = graph

        # 计算损失
        loss = dgi_loss_fn(encoder, graph_aug, spatial_aware_corruption)

        # 反向传播
        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step(epoch)

        avg_loss = loss.item()

        # 日志
        if epoch % 50 == 0:
            print(f"Epoch {epoch:04d} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 保存最优模型
        if avg_loss < min_loss:
            min_loss = avg_loss
            patience = 0

            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'loss_fn_state_dict': dgi_loss_fn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': min_loss
            }, model_file)

            # 保存嵌入和注意力
            encoder.eval()
            with torch.no_grad():
                z = encoder(graph)

                # 保存嵌入（使用子目录）
                emb_file = os.path.join(embedding_subdir, f'{args.model_name}_Embed_X')
                with gzip.open(emb_file, 'wb') as fp:
                    pickle.dump(z.cpu().numpy(), fp)

                # 保存注意力（兼容原有后处理格式）
                att_raw = encoder.layer_attentions_raw[-1].cpu().numpy()  # 最后一层
                att_norm = encoder.layer_attentions_norm[-1].cpu().numpy()

                # 格式：[idx_l1, att_l1, att2_raw, att1_raw, att_l2, idx_l2]
                bundle = [
                    graph.edge_index.cpu().numpy(),  # idx_l1
                    att_norm,  # att_l1 (normalized)
                    att_raw,  # att2_raw (unnormalized) - 用于CCC分数
                    att_raw,  # att1_raw
                    att_norm,  # att_l2 (normalized)
                    graph.edge_index.cpu().numpy()  # idx_l2
                ]

                # 保存注意力（使用子目录）
                att_file = os.path.join(embedding_subdir, f'{args.model_name}_attention')
                with gzip.open(att_file, 'wb') as fp:
                    pickle.dump(bundle, fp)
        else:
            patience += 1
        # 早停
        if epoch > min_stop and patience > patience_limit:
            print(f"\n✅ 早停于 epoch {epoch}, best_loss = {min_loss:.6f}")
            break
    # 加载最优模型
    ckpt = torch.load(model_file, map_location=device)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    encoder.eval()

    print(f"\n{'=' * 60}")
    print(f"✅ 训练完成！最终损失: {min_loss:.6f}")
    print(f"{'=' * 60}")

    return encoder


# ==================== 7. 辅助函数 ====================

def get_graphs(training_data_base, expression_matrix):
    """加载图数据（与原版兼容）"""
    if expression_matrix is None:
        raise ValueError("必须提供 expression_matrix 参数！")

    p = f"{training_data_base}"
    if not os.path.exists(p):
        raise FileNotFoundError(f"未找到图文件: {p}")

    with gzip.open(p, 'rb') as f:
        row_col, edge_weight, lig_rec, num_cell = pickle.load(f)

    edge_index = torch.tensor(np.array(row_col), dtype=torch.long).T
    edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)

    # 处理节点特征
    X = np.asarray(expression_matrix, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 标准化
    col_mean = X.mean(axis=0, keepdims=True)
    col_std = X.std(axis=0, keepdims=True)
    col_std[col_std < 1e-6] = 1.0
    X = (X - col_mean) / col_std

    X_data = torch.tensor(X, dtype=torch.float)

    graph = Data(x=X_data, edge_index=edge_index, edge_attr=edge_attr)

    num_feature = X_data.shape[1]
    edge_dim = edge_attr.size(1)
    rel_vocab = int(edge_attr[:, 2].max().item()) + 1 if edge_dim >= 3 else 1

    print(f"\n✅ 图数据加载完成")
    print(f"   节点数: {num_cell}, 特征维度: {num_feature}")
    print(f"   边数: {graph.edge_index.size(1)}, 关系类型: {rel_vocab}")

    return graph, num_feature, edge_dim, rel_vocab

