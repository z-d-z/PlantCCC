import os
import numpy as np
import pandas as pd
import random
import torch
import torch.backends.cudnn as cudnn
import scanpy as sc

# 使用整合后的data_pre模块
from data_pre import DataProcessor, PreprocessorPlus
# 新增：引入后处理和可视化模块
from output_postprocessor import PlantCCCPostProcessor
from output_visualizer import PlantCCCVisualizer

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          📋 配置参数区 - 集中修改                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ==================== 数据路径配置 ====================
# ✅ 修改：直接指定 .h5ad 文件路径
H5AD_FILE_PATH = "../du-juan/SA-IN11/IN7.h5ad"  # 👈 修改为你的 .h5ad 文件路径
DATA_NAME = 'IN7'  # 数据集名称

# 聚类结果文件路径（包含细胞类型注释）
# ✅ 如果聚类信息已在 h5ad 文件中，可设为 None
CLUSTER_H5AD_PATH = "../du-juan/SA-IN11/IN7.h5ad"

# L-R配对数据库路径
DATABASE_PATH = "LR_pair_Ptrichocarpa_多_formatted.csv"

# ==================== 输出路径配置 ====================
OUTPUT_ROOT = "output/IN7"  # 所有输出的根目录
MODEL_NAME = "IN7"  # 模型名称标识

# ==================== 数据加载模式 ====================
# ✅ 新增：设置为 "h5ad" 模式
DATA_LOADING_MODE = "h5ad"  # "h5ad" 或 "Visium"

# ==================== 空间坐标检查 ====================
# ✅ 新增：指定空间坐标在 h5ad 中的键名
SPATIAL_KEY = "spatial"  # adata.obsm['spatial'] 的键名

# ==================== 模型选择 ====================
USE_OPTIMIZED_MODEL = True  # True=使用植物优化版, False=使用原版

# ==================== 数据处理参数 ====================
# 基因表达增强参数
SPATIAL_TYPE = "KDTree"  # 空间邻域算法
ADJACENT_WEIGHT = 0.4  # 邻域基因表达权重
NEIGHBOUR_K = 6  # K近邻数量

# PCA降维参数
PCA_N_COMPS = 200  # 主成分数量

# ==================== PlantCCC预处理参数 ====================
if USE_OPTIMIZED_MODEL:
    # 植物优化版参数
    DISTANCE_MULTIPLIER = 1.5  # 空间邻域距离倍数（更大邻域）
    THRESHOLD_GENE_EXP = 95  # 基因表达阈值百分位（更宽松）
    BLOCK_AUTOCRINE = 1  # 是否阻止自分泌（0=允许, 1=禁止）
else:
    # 原版参数
    DISTANCE_MULTIPLIER = 1.5
    THRESHOLD_GENE_EXP = 98
    BLOCK_AUTOCRINE = 1

# ==================== 模型训练参数 ====================
if USE_OPTIMIZED_MODEL:
    # 植物优化版训练参数
    HIDDEN_DIM = 256  # 隐藏层维度
    ATTENTION_HEADS = 4  # 注意力头数
    NUM_LAYERS = 3  # GAT层数
    DROPOUT = 0.1  # Dropout率
    REL_EMB_DIM = 16  # 关系嵌入维度

    NUM_EPOCH = 500  # 训练轮数
    LEARNING_RATE = 2e-4  # 学习率
    DGI_TAU = 0.4  # DGI对比学习温度
    PATIENCE_LIMIT = 200  # 早停耐心值
    MIN_STOP = 500  # 最小训练轮数
else:
    # 原版训练参数
    HIDDEN_DIM = 256
    ATTENTION_HEADS = 1
    NUM_LAYERS = 2
    DROPOUT = 0.1
    REL_EMB_DIM = 8

    NUM_EPOCH = 2000
    LEARNING_RATE = 1e-3
    DGI_TAU = 0.3
    PATIENCE_LIMIT = 300
    MIN_STOP = 800

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
print(f"📁 数据模式: {DATA_LOADING_MODE}")
if DATA_LOADING_MODE == "h5ad":
    print(f"📄 H5AD文件: {H5AD_FILE_PATH}")
print(f"🧬 L-R数据库: {DATABASE_PATH}")
print(f"📍 聚类文件: {CLUSTER_H5AD_PATH if CLUSTER_H5AD_PATH else '(使用h5ad内置)'}")
print(f"\n🤖 模型模式: {'植物优化版 (Plant-PlantCCC)' if USE_OPTIMIZED_MODEL else '原版 (PlantCCC)'}")
print(f"💾 输出目录: {OUTPUT_ROOT}")
print(f"🏷️  模型名称: {MODEL_NAME}")

if USE_OPTIMIZED_MODEL:
    print(f"\n⚙️  优化参数:")
    print(f"   - 空间邻域倍数: {DISTANCE_MULTIPLIER}x")
    print(f"   - 表达阈值: {THRESHOLD_GENE_EXP}%")
    print(f"   - 自分泌: {'允许' if BLOCK_AUTOCRINE == 0 else '禁止'}")
    print(f"   - 注意力头数: {ATTENTION_HEADS}")
    print(f"   - 网络层数: {NUM_LAYERS}")
    print(f"   - 关系嵌入维度: {REL_EMB_DIM}")

# ==================== Step 1: 数据加载 ====================
print("\n" + "=" * 70)
print("Step 1: 数据加载")
print("=" * 70)

if DATA_LOADING_MODE == "h5ad":
    # ✅ 新增：直接从 .h5ad 文件加载
    print(f"正在从 .h5ad 文件加载数据: {H5AD_FILE_PATH}")
    adata = sc.read_h5ad(H5AD_FILE_PATH)
    adata.var_names_make_unique()

    # 检查空间坐标
    if SPATIAL_KEY not in adata.obsm:
        raise ValueError(f"❌ 错误: adata.obsm 中未找到 '{SPATIAL_KEY}' 键！")

    print(f"✅ 数据加载完成: {adata.shape[0]} 细胞, {adata.shape[1]} 基因")
    print(f"✅ 空间坐标形状: {adata.obsm[SPATIAL_KEY].shape}")

    # ✅ 新增：检查并转换稀疏矩阵
    import scipy.sparse as sp

    if sp.issparse(adata.X):
        print(f"⚠️  检测到稀疏矩阵格式: {type(adata.X).__name__}")
        print(f"   矩阵大小: {adata.X.shape}")
        print(f"   非零元素: {adata.X.nnz:,}")
        sparsity = 100 * (1 - adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]))
        print(f"   稀疏度: {sparsity:.2f}%")
        print("   正在转换为稠密矩阵...")
        adata.X = adata.X.toarray()
        print("✅ 转换完成")

    # 检查是否有聚类信息
    if 'cluster' in adata.obs.columns or 'leiden' in adata.obs.columns or 'louvain' in adata.obs.columns:
        print("✅ 检测到 h5ad 文件中包含聚类信息")
    else:
        print("⚠️  警告: h5ad 文件中未检测到聚类信息，将使用外部聚类文件")

    # ==================== 🔧 修复: 检查并缩放空间坐标 ====================
    coords = adata.obsm[SPATIAL_KEY]
    coord_range_x = coords[:, 0].max() - coords[:, 0].min()
    coord_range_y = coords[:, 1].max() - coords[:, 1].min()

    print(f"\n📏 空间坐标范围检查:")
    print(f"   X: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}] (范围: {coord_range_x:.2f})")
    print(f"   Y: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}] (范围: {coord_range_y:.2f})")

    # 如果坐标范围太小，缩放到标准Visium范围
    if coord_range_x < 100 or coord_range_y < 100:
        print(f"\n⚠️  检测到坐标范围过小，正在自动缩放...")

        # Visium标准坐标范围约为 0-30000
        target_range = 20000
        scale_factor = max(target_range / coord_range_x, target_range / coord_range_y)

        # 缩放坐标
        adata.obsm[SPATIAL_KEY] = coords * scale_factor

        # 验证
        coords_new = adata.obsm[SPATIAL_KEY]
        print(f"✅ 坐标已缩放 {scale_factor:.2f} 倍")
        print(f"   新X范围: [{coords_new[:, 0].min():.2f}, {coords_new[:, 0].max():.2f}]")
        print(f"   新Y范围: [{coords_new[:, 1].min():.2f}, {coords_new[:, 1].max():.2f}]")
    else:
        print("✅ 坐标范围正常，无需缩放")

elif DATA_LOADING_MODE == "Visium":
    # 原始 Visium 加载方式
    processor = DataProcessor(save_path=OUTPUT_ROOT, use_gpu=True)
    adata = processor.get_adata(
        platform="Visium",
        data_path=RAW_DATA_ROOT,
        data_name=DATA_NAME
    )
    adata.var_names_make_unique()
    print(f"✅ 数据加载完成: {adata.shape[0]} 细胞, {adata.shape[1]} 基因")

else:
    raise ValueError(f"不支持的数据加载模式: {DATA_LOADING_MODE}")

# ==================== Step 2: 基因表达增强 ====================
print("\n" + "=" * 70)
print("Step 2: 基因表达增强")
print("=" * 70)

processor = DataProcessor(save_path=OUTPUT_ROOT, use_gpu=True)

# 基因表达增强
adata = processor.get_augment(
    adata,
    spatial_type=SPATIAL_TYPE,
    adjacent_weight=ADJACENT_WEIGHT,
    neighbour_k=NEIGHBOUR_K
)

# 用于图构建的增强表达矩阵
data = processor.data_preprocess_ccc(adata)

print(f"✅ 增强表达矩阵形状: {data.shape}")

# ==================== Step 3: PlantCCC 预处理 ====================
print("\n" + "=" * 70)
print("Step 3: PlantCCC 预处理 (构建空间邻接图与L-R匹配)")
print("=" * 70)

preprocessor = PreprocessorPlus(
    data_name=DATA_NAME,
    adata=adata,
    enhanced_expression=data,
    cluster_h5ad_path=CLUSTER_H5AD_PATH,  # 可以是 None
    base_distance_multiplier=DISTANCE_MULTIPLIER,
    database_path=DATABASE_PATH,
    threshold_gene_exp=THRESHOLD_GENE_EXP,
    block_autocrine=BLOCK_AUTOCRINE,
    data_to=PATH_CONF['input_graph'],
    metadata_to=PATH_CONF['metadata']
)

preprocessor.run()

# ==================== Step 4: 准备训练特征 ====================
print("\n" + "=" * 70)
print("Step 4: 准备 GAT 节点特征")
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

# ==================== Step 5: GAT-DGI 模型训练 ====================
print("\n" + "=" * 70)
print(f"Step 5: GAT-DGI 模型训练 ({'优化版' if USE_OPTIMIZED_MODEL else '原版'})")
print("=" * 70)

if USE_OPTIMIZED_MODEL:
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

else:
    # ===== 使用原版模型 =====
    from train_GAT_DGI import CCCTrainer

    trainer = CCCTrainer(
        data_name=DATA_NAME,
        model_name=MODEL_NAME,
        expression_matrix=data_ccc,

        training_data=PATH_CONF['input_graph'],
        model_path=PATH_CONF['model'],
        embedding_path=PATH_CONF['embedding'],

        num_epoch=NUM_EPOCH,
        hidden=HIDDEN_DIM,
        heads=ATTENTION_HEADS,
        lr_rate=LEARNING_RATE,
        dropout=DROPOUT,
        seed=RANDOM_SEED,
        rel_emb_dim=REL_EMB_DIM,

        patience_limit=PATIENCE_LIMIT,
        min_stop=MIN_STOP,
        dgi_tau=DGI_TAU,
        grad_clip=GRAD_CLIP,
    )

    trainer.print_configuration()
    model = trainer.train()

print(f"✅ 训练完成，嵌入保存至: {PATH_CONF['embedding']}")

# ==================== Step 6: 后处理 ====================
print("\n" + "=" * 70)
print("Step 6: 后处理 (解析Attention权重生成CCC列表)")
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

# ==================== Step 7: 可视化 ====================
print("\n" + "=" * 70)
print("Step 7: 自动化可视化 (生成 HTML 交互图)")
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
print(f"\n🔧 使用的模型: {'优化版 (Plant-PlantCCC)' if USE_OPTIMIZED_MODEL else '原版 (PlantCCC)'}")
print("=" * 70)