import os
import numpy as np
import pandas as pd
import random
import torch
import torch.backends.cudnn as cudnn
# 使用整合后的data_pre模块
from data_pre import DataProcessor, PreprocessorPlus
# 新增：引入后处理和可视化模块
from output_postprocessor import PlantCCCPostProcessor
from output_visualizer import PlantCCCVisualizer

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          📋 配置参数区 - 集中修改                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ==================== 数据路径配置 ====================
RAW_DATA_ROOT = "data"  # 原始数据根目录
DATA_NAME = 'IN4'  # 数据集名称

# 聚类结果文件路径（包含细胞类型注释）
CLUSTER_H5AD_PATH = "data/IN4/outs_4_with_clusters.h5ad"

# L-R配对数据库路径
DATABASE_PATH = "LR_pair_Ptrichocarpa.csv"

# ==================== 输出路径配置 ====================
OUTPUT_ROOT = "output/heng_4"  # 所有输出的根目录
MODEL_NAME = "heng_4"  # 模型名称标识
# ==================== 数据处理参数 ====================
# 数据平台类型
PLATFORM = "Visium"

# 基因表达增强参数
SPATIAL_TYPE = "KDTree"  # 空间邻域算法
ADJACENT_WEIGHT = 0.4  # 邻域基因表达权重
NEIGHBOUR_K = 6  # K近邻数量

# PCA降维参数
PCA_N_COMPS = 200  # 主成分数量

# ==================== 预处理参数 ====================

DISTANCE_MULTIPLIER = 2  # 空间邻域距离倍数（更大邻域）
THRESHOLD_GENE_EXP = 90  # 基因表达阈值百分位（更宽松）
BLOCK_AUTOCRINE = 1  # 是否阻止自分泌（0=允许, 1=禁止）

# ==================== 模型训练参数 ====================

HIDDEN_DIM = 256  # 隐藏层维度
ATTENTION_HEADS = 4  # 注意力头数
NUM_LAYERS = 3  # GAT层数
DROPOUT = 0.1  # Dropout率
REL_EMB_DIM = 16  # 关系嵌入维度

NUM_EPOCH = 1000  # 训练轮数
LEARNING_RATE = 2e-4  # 学习率
DGI_TAU = 0.4  # DGI对比学习温度
PATIENCE_LIMIT = 200  # 早停耐心值
MIN_STOP = 500  # 最小训练轮数

# 通用训练参数
RANDOM_SEED = 36  # 随机种子
GRAD_CLIP = 1.0  # 梯度裁剪阈值

# ==================== 后处理与可视化参数 ====================
TOP_PERCENT = 20  # 保留top百分比的CCC
TOP_EDGE_COUNT = 10000  # 可视化时显示的最大边数
FILTER_THRESHOLD = 0  # 过滤阈值
SORT_BY_ATTENTION = 1  # 是否按attention分数排序

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          🚀 主程序执行区 - 无需修改                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ==================== 0. 全局配置与环境设置 ====================
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

# 随机种子固定
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
cudnn.deterministic = True
cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# ==================== 1. 自动构建输出目录结构 ====================
PATH_CONF = {
    'metadata': os.path.join(OUTPUT_ROOT, 'metadata'),
    'input_graph': os.path.join(OUTPUT_ROOT, 'input_graph'),
    'embedding': os.path.join(OUTPUT_ROOT, 'embedding_data'),
    'model': os.path.join(OUTPUT_ROOT, 'model'),
    'vis_output': os.path.join(OUTPUT_ROOT, 'visualization')
}

for p in PATH_CONF.values():
    os.makedirs(p, exist_ok=True)

# ==================== 打印配置信息 ====================
print("\n" + "╔" + "═" * 68 + "╗")
print("║" + " " * 20 + "🔧 PlantCCC 配置信息" + " " * 27 + "║")
print("╚" + "═" * 68 + "╝")
print(f"\n📊 数据集: {DATA_NAME}")
print(f"📁 数据路径: {RAW_DATA_ROOT}")
print(f"🧬 L-R数据库: {DATABASE_PATH}")
print(f"📍 聚类文件: {CLUSTER_H5AD_PATH}")
print(f"\n🤖 模型模式: {'植物优化版 (Plant-PlantCCC)'}")
print(f"💾 输出目录: {OUTPUT_ROOT}")
print(f"🏷️  模型名称: {MODEL_NAME}")
print(f"\n⚙️  优化参数:")
print(f"   - 空间邻域倍数: {DISTANCE_MULTIPLIER}x")
print(f"   - 表达阈值: {THRESHOLD_GENE_EXP}%")
print(f"   - 自分泌: {'允许' if BLOCK_AUTOCRINE == 0 else '禁止'}")
print(f"   - 注意力头数: {ATTENTION_HEADS}")
print(f"   - 网络层数: {NUM_LAYERS}")
print(f"   - 关系嵌入维度: {REL_EMB_DIM}")

# ==================== Step 1: 数据加载与增强 ====================
print("\n" + "=" * 70)
print("Step 1: 数据加载与基因表达增强")
print("=" * 70)

processor = DataProcessor(save_path=OUTPUT_ROOT, use_gpu=True)

# 加载数据
adata = processor.get_adata(
    platform=PLATFORM,
    data_path=RAW_DATA_ROOT,
    data_name=DATA_NAME
)
adata.var_names_make_unique()

# 基因表达增强
adata = processor.get_augment(
    adata,
    spatial_type=SPATIAL_TYPE,
    adjacent_weight=ADJACENT_WEIGHT,
    neighbour_k=NEIGHBOUR_K
)

# 用于图构建的增强表达矩阵
data = processor.data_preprocess_ccc(adata)

print(f"✅ 数据加载完成: {adata.shape[0]} 细胞, {adata.shape[1]} 基因")
print(f"✅ 增强表达矩阵形状: {data.shape}")

# ==================== Step 2: PlantCCC 预处理 ====================
print("\n" + "=" * 70)
print("Step 2: PlantCCC 预处理 (构建空间邻接图与L-R匹配)")
print("=" * 70)

preprocessor = PreprocessorPlus(
    data_name=DATA_NAME,
    adata=adata,
    enhanced_expression=data,
    cluster_h5ad_path=CLUSTER_H5AD_PATH,
    base_distance_multiplier=DISTANCE_MULTIPLIER,
    database_path=DATABASE_PATH,
    threshold_gene_exp=THRESHOLD_GENE_EXP,
    block_autocrine=BLOCK_AUTOCRINE,
    data_to=PATH_CONF['input_graph'],
    metadata_to=PATH_CONF['metadata']
)

preprocessor.run()

# ==================== Step 3: 准备训练特征 ====================
print("\n" + "=" * 70)
print("Step 3: 准备 GAT 节点特征")
print("=" * 70)

data_ccc = processor.data_preprocess_identify(adata, pca_n_comps=PCA_N_COMPS)

# 数据一致性验证
print("\n--- 数据一致性验证 ---")
n_cells_adata = len(adata.obs_names)
n_cells_feature = data_ccc.shape[0]

assert n_cells_feature == n_cells_adata, \
    f"❌ 数据不一致: 节点特征行数 ({n_cells_feature}) != adata细胞数 ({n_cells_adata})"

print(f"✅ 细胞数量一致: {n_cells_adata}")
print(f"✅ 节点特征维度: {data_ccc.shape[1]}")

# ==================== Step 4: GAT-DGI 模型训练 ====================
print("\n" + "=" * 70)
print(f"Step 4: GAT-DGI 模型训练 ({'优化版' })")
print("=" * 70)


# ===== 使用优化版模型 =====
from CCC_get_plant_optimized import train_plant_PlantCCC, get_graphs
import types

# 构建图数据路径
training_data_path = os.path.join(
    PATH_CONF['input_graph'],
    DATA_NAME,
    f"{DATA_NAME}_adjacency_records"
)

# 加载图
graph, num_feature, edge_dim, rel_vocab = get_graphs(
    training_data_path,
    expression_matrix=data_ccc
)

# 构造参数对象
args = types.SimpleNamespace(
    data_name=DATA_NAME,
    model_name=MODEL_NAME,

    hidden=HIDDEN_DIM,
    heads=ATTENTION_HEADS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    rel_emb_dim=REL_EMB_DIM,

    num_epoch=NUM_EPOCH,
    lr_rate=LEARNING_RATE,
    dgi_tau=DGI_TAU,
    patience_limit=PATIENCE_LIMIT,
    min_stop=MIN_STOP,

    model_path=PATH_CONF['model'],
    embedding_path=PATH_CONF['embedding'],
)

print("\n===== 训练配置 =====")
print(f"  训练轮数: {args.num_epoch}")
print(f"  学习率: {args.lr_rate}")
print(f"  DGI温度: {args.dgi_tau}")
print(f"  早停耐心: {args.patience_limit}")
print("=" * 20 + "\n")

# 训练
model = train_plant_PlantCCC(args, graph, num_feature, edge_dim, rel_vocab)


print(f"✅ 训练完成，嵌入保存至: {PATH_CONF['embedding']}")

# ==================== Step 5: 后处理 ====================
print("\n" + "=" * 70)
print("Step 5: 后处理 (解析Attention权重生成CCC列表)")
print("=" * 70)

post_processor = PlantCCCPostProcessor(
    data_name=DATA_NAME,
    model_name=MODEL_NAME,
    embedding_path=PATH_CONF['embedding'],
    metadata_from=PATH_CONF['metadata'],
    data_from=PATH_CONF['input_graph'],
    output_path=PATH_CONF['vis_output'],
    top_percent=TOP_PERCENT
)

post_processor.run()

# ==================== Step 6: 可视化 ====================
print("\n" + "=" * 70)
print("Step 6: 自动化可视化 (生成 HTML 交互图)")
print("=" * 70)

top_ccc_csv = os.path.join(
    PATH_CONF['vis_output'],
    DATA_NAME,
    f"{MODEL_NAME}_top{TOP_PERCENT}percent.csv"
)

visualizer = PlantCCCVisualizer(
    data_name=DATA_NAME,
    model_name=MODEL_NAME,
    top_edge_count=TOP_EDGE_COUNT,
    top_ccc_file=top_ccc_csv,
    metadata_from=os.path.join(PATH_CONF['metadata'], DATA_NAME),
    output_path=PATH_CONF['vis_output'],
    filter=FILTER_THRESHOLD,
    sort_by_attentionScore=SORT_BY_ATTENTION
)

visualizer.run()

# ==================== 完成 ====================
print("\n" + "=" * 70)
print(f"🎉 全部流程执行完毕！")
print(f"📊 结果文件位于: {PATH_CONF['vis_output']}")
print(f"   - CCC列表 (CSV): {os.path.basename(top_ccc_csv)}")
print(f"   - 可视化网页 (HTML): *_mygraph.html, *_component_plot.html 等")
print(f"\n🔧 使用的模型: {'优化版 (Plant-PlantCCC)'}")
print("=" * 70)