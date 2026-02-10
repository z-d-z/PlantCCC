
import os
import math
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from pathlib import Path
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, KDTree, BallTree
from scipy.sparse import csr_matrix, issparse
from scipy import sparse
import scipy.sparse as sp
from tqdm import tqdm
from collections import defaultdict
import gzip
import pickle
import gc

# ==================== 1. 数据读取函数 (从PlantST整合) ====================

def read_10X_Visium(path,
                    genome=None,
                    count_file='filtered_feature_bc_matrix.h5',
                    library_id=None,
                    load_images=True,
                    source_image_path=None):
    """
    读取10X Visium空间转录组数据

    参数:
        path: 数据目录路径
        genome: 基因组版本
        count_file: 计数矩阵文件名
        library_id: 库ID
        load_images: 是否加载图像
        source_image_path: 图像源路径

    返回:
        adata: AnnData对象
    """
    adata = sc.read_visium(path,
                           genome=genome,
                           count_file=count_file,
                           library_id=library_id,
                           load_images=load_images,
                           source_image_path=source_image_path)
    adata.var_names_make_unique()
    return adata

def ReadOldST(path, count_matrix_file=None, spatial_file=None, use_quality='hires'):
    """
    读取旧版ST格式数据

    参数:
        path: 数据目录路径
        count_matrix_file: 计数矩阵文件
        spatial_file: 空间坐标文件
        use_quality: 图像质量

    返回:
        adata: AnnData对象
    """
    # 查找计数矩阵文件
    if count_matrix_file is None:
        for f in os.listdir(path):
            if f.endswith('_stdata.tsv') or f.endswith('_stdata.tsv.gz'):
                count_matrix_file = f
                break

    # 读取计数矩阵
    count_path = os.path.join(path, count_matrix_file)
    if count_path.endswith('.gz'):
        counts = pd.read_csv(count_path, sep='\t', compression='gzip', index_col=0)
    else:
        counts = pd.read_csv(count_path, sep='\t', index_col=0)

    counts = counts.T  # 转置为 cells x genes
    adata = anndata.AnnData(counts)

    # 读取空间坐标
    if spatial_file is None:
        for f in os.listdir(path):
            if 'spot' in f.lower() and f.endswith('.tsv'):
                spatial_file = f
                break

    if spatial_file:
        spatial_path = os.path.join(path, spatial_file)
        spatial = pd.read_csv(spatial_path, sep='\t', index_col=0)

        # 匹配barcode
        common = adata.obs_names.intersection(spatial.index)
        adata = adata[common, :].copy()
        spatial = spatial.loc[common, :]

        # 提取坐标
        if 'x' in spatial.columns and 'y' in spatial.columns:
            adata.obsm['spatial'] = spatial[['x', 'y']].values
        elif 'imagerow' in spatial.columns and 'imagecol' in spatial.columns:
            adata.obsm['spatial'] = spatial[['imagecol', 'imagerow']].values

    adata.var_names_make_unique()
    return adata

def read_merfish(path):
    """读取MERFISH数据"""
    counts_file = None
    meta_file = None

    for f in os.listdir(path):
        if 'count' in f.lower() and f.endswith('.csv'):
            counts_file = f
        if 'meta' in f.lower() and f.endswith('.csv'):
            meta_file = f

    counts = pd.read_csv(os.path.join(path, counts_file), index_col=0)
    adata = anndata.AnnData(counts)

    if meta_file:
        meta = pd.read_csv(os.path.join(path, meta_file), index_col=0)
        common = adata.obs_names.intersection(meta.index)
        adata = adata[common, :].copy()
        meta = meta.loc[common, :]

        # 查找坐标列
        x_col = None
        y_col = None
        for c in meta.columns:
            if 'x' in c.lower() and x_col is None:
                x_col = c
            if 'y' in c.lower() and y_col is None:
                y_col = c

        if x_col and y_col:
            adata.obsm['spatial'] = meta[[x_col, y_col]].values

    adata.var_names_make_unique()
    return adata


def read_SlideSeq(path):
    """读取SlideSeq数据"""
    counts_file = None
    pos_file = None

    for f in os.listdir(path):
        if f.endswith('.csv') or f.endswith('.txt'):
            if 'count' in f.lower() or 'expression' in f.lower():
                counts_file = f
            if 'position' in f.lower() or 'location' in f.lower():
                pos_file = f

    counts = pd.read_csv(os.path.join(path, counts_file), index_col=0)
    adata = anndata.AnnData(counts.T)

    if pos_file:
        pos = pd.read_csv(os.path.join(path, pos_file), index_col=0)
        common = adata.obs_names.intersection(pos.index)
        adata = adata[common, :].copy()
        pos = pos.loc[common, :]
        adata.obsm['spatial'] = pos.values[:, :2]

    adata.var_names_make_unique()
    return adata


def read_stereoSeq(path):
    """读取stereoSeq数据"""
    for f in os.listdir(path):
        if f.endswith('.h5ad'):
            adata = sc.read_h5ad(os.path.join(path, f))
            break
        elif f.endswith('.gem') or f.endswith('.gem.gz'):
            gem_path = os.path.join(path, f)
            if f.endswith('.gz'):
                gem = pd.read_csv(gem_path, sep='\t', compression='gzip', comment='#')
            else:
                gem = pd.read_csv(gem_path, sep='\t', comment='#')

            # 创建barcode
            gem['barcode'] = gem['x'].astype(str) + '_' + gem['y'].astype(str)

            # pivot to cells x genes
            counts = gem.pivot_table(index='barcode', columns='geneID', values='MIDCount', fill_value=0)
            adata = anndata.AnnData(counts)

            # 添加空间坐标
            coords = gem.groupby('barcode')[['x', 'y']].first()
            adata.obsm['spatial'] = coords.loc[adata.obs_names, :].values
            break

    adata.var_names_make_unique()
    return adata


# ==================== 2. 基因表达增强函数 (从augment.py整合，移除形态学特征) ====================

def cal_spatial_weight(data, spatial_type, spatial_k):
    """
    计算空间邻域权重矩阵

    参数:
        data: 空间坐标数组 (N x 2)
        spatial_type: 算法类型 ("NearestNeighbors", "KDTree", "BallTree")
        spatial_k: 最近邻数量

    返回:
        spatial_weight: 空间邻接矩阵 (N x N)
    """
    if spatial_type == "NearestNeighbors":
        nbrs = NearestNeighbors(n_neighbors=spatial_k + 1, algorithm='ball_tree').fit(data)
        _, indices = nbrs.kneighbors(data)
    elif spatial_type == "KDTree":
        tree = KDTree(data, leaf_size=2)
        _, indices = tree.query(data, k=spatial_k + 1)
    elif spatial_type == "BallTree":
        tree = BallTree(data, leaf_size=2)
        _, indices = tree.query(data, k=spatial_k + 1)
    else:
        raise ValueError(f"不支持的spatial_type: {spatial_type}")

    # 去除每个点自身的索引，只保留其他的邻居
    indices = indices[:, 1:]

    # 初始化全零矩阵
    spatial_weight = np.zeros((data.shape[0], data.shape[0]))
    print("Starting to fill spatial_weight matrix.")

    for i in range(indices.shape[0]):
        ind = indices[i]
        for j in ind:
            spatial_weight[i][j] = 1

    return spatial_weight


def cal_gene_weight(data, n_components, gene_dist_type):
    """
    计算基因表达相关性矩阵

    参数:
        data: 基因表达矩阵 (cells x genes)
        n_components: PCA降维后的维度数
        gene_dist_type: 距离度量类型

    返回:
        gene_correlation: 基因相关性矩阵 (N x N)
    """
    if isinstance(data, csr_matrix):
        data = data.toarray()

    if data.shape[1] > 500:
        pca = PCA(n_components=n_components)
        data = pca.fit_transform(data)
        gene_correlation = 1 - pairwise_distances(data, metric=gene_dist_type)
    else:
        gene_correlation = 1 - pairwise_distances(data, metric=gene_dist_type)

    return gene_correlation


def cal_weight_matrix_no_morphology(adata, spatial_type, spatial_k,
                                    gb_dist_type="correlation", n_components=100):
    """
    计算权重矩阵（不使用形态学特征）

    参数:
        adata: AnnData对象
        spatial_type: 空间算法类型 ("LinearRegress", "NearestNeighbors", "KDTree", "BallTree")
        spatial_k: 空间邻域大小
        gb_dist_type: 基因表达相关性计算的距离类型
        n_components: PCA组件数量

    返回:
        adata: 添加了权重矩阵的AnnData对象
    """
    # 计算物理距离矩阵
    physical_distance = cal_spatial_weight(
        adata.obsm['spatial'],
        spatial_k=spatial_k,
        spatial_type=spatial_type
    )

    print("Physical distance calculating Done!")
    print(f"The number of nearest tie neighbors in physical distance is: {physical_distance.sum() / adata.shape[0]}")

    # 基因表达相关性计算
    gene_correlation = cal_gene_weight(
        data=adata.X.copy(),
        gene_dist_type=gb_dist_type,
        n_components=n_components
    )
    gene_correlation[gene_correlation < 0] = 0
    print("Gene correlation calculating Done!")

    # 不使用形态学特征，直接使用基因相关性和物理距离
    adata.obsm["weights_matrix_all"] = gene_correlation * physical_distance
    adata.obsm["gene_correlation"] = gene_correlation
    print("The weight result is added to adata.obsm['weights_matrix_all'] !")

    return adata


def find_adjacent_spot(adata, use_data, neighbour_k, verbose=False):
    """
    找到每个spot的邻近spot并计算加权基因表达

    参数:
        adata: AnnData对象
        use_data: 使用的数据类型 ("raw" 或 obsm中的键名)
        neighbour_k: 每个spot的邻域大小
        verbose: 是否输出详细信息

    返回:
        adata: 添加了邻近数据的AnnData对象
    """
    # 判断矩阵结构
    if use_data == "raw":
        if isinstance(adata.X, csr_matrix):
            gene_matrix = adata.X.toarray()
        elif isinstance(adata.X, np.ndarray):
            gene_matrix = adata.X
        elif isinstance(adata.X, pd.DataFrame):
            gene_matrix = adata.X.values
        else:
            raise ValueError(f"{type(adata.X)} is not a valid type.")
    else:
        gene_matrix = adata.obsm[use_data]

    weights_list = []  # 存储每个spot的邻近权重
    final_coordinates = []  # 存储加权后的基因表达数据

    # 打印进度条
    with tqdm(total=len(adata), desc="Find adjacent spots of each spot",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:

        # 遍历所有spots
        for i in range(adata.shape[0]):
            # 找到临近的spot (权重最大的邻近spots的索引)
            current_spot = adata.obsm['weights_matrix_all'][i].argsort()[-neighbour_k:][:neighbour_k - 1]

            # 获取选定邻近spots的权重
            spot_weight = adata.obsm['weights_matrix_all'][i][current_spot]

            # 获取这些邻近spots的基因表达数据
            spot_matrix = gene_matrix[current_spot]

            # 权重加权和计算
            if spot_weight.sum() > 0:
                # 权重归一化
                spot_weight_scaled = (spot_weight / spot_weight.sum())
                weights_list.append(spot_weight_scaled)

                # 加权平均的基因表达数据
                spot_matrix_scaled = np.multiply(spot_weight_scaled.reshape(-1, 1), spot_matrix)
                spot_matrix_final = np.sum(spot_matrix_scaled, axis=0)
            else:
                spot_matrix_final = np.zeros(gene_matrix.shape[1])
                weights_list.append(np.zeros(len(current_spot)))

            final_coordinates.append(spot_matrix_final)
            pbar.update(1)

    adata.obsm['adjacent_data'] = np.array(final_coordinates)
    if verbose:
        adata.obsm['adjacent_weight'] = np.array(weights_list)

    return adata


def augment_gene_data(adata, adjacent_weight):
    """
    根据邻近点的信息增强基因表达数据

    参数:
        adata: AnnData对象
        adjacent_weight: 邻近数据的权重因子

    返回:
        adata: 添加了增强基因表达的AnnData对象
    """
    if isinstance(adata.X, np.ndarray):
        adata.obsm["augment_gene_data"] = adata.X + adjacent_weight * adata.obsm["adjacent_data"].astype(float)
    elif isinstance(adata.X, csr_matrix):
        adata.obsm["augment_gene_data"] = adata.X.toarray() + adjacent_weight * adata.obsm["adjacent_data"].astype(
            float)

    return adata


def augment_adata_no_morphology(adata, spatial_type, spatial_k, adjacent_weight,
                                gb_dist_type="correlation", n_components=100,
                                use_data="raw", neighbour_k=6):
    """
    参数:
        adata: AnnData对象
        spatial_type: 空间算法类型
        spatial_k: 空间邻域大小
        adjacent_weight: 邻近点在增强过程中的占比
        gb_dist_type: 基因表达距离的度量方法
        n_components: PCA组件数量
        use_data: 指定增强基因表达时使用的基因数据
        neighbour_k: 每个spot的邻域大小

    返回:
        adata: 增强后的AnnData对象
    """
    # 计算权重矩阵（不使用形态学特征）
    adata = cal_weight_matrix_no_morphology(
        adata,
        gb_dist_type=gb_dist_type,
        n_components=n_components,
        spatial_k=spatial_k,
        spatial_type=spatial_type,
    )

    # 找到邻近点并计算加权基因表达
    adata = find_adjacent_spot(
        adata,
        use_data=use_data,
        neighbour_k=neighbour_k
    )

    # 增强基因表达数据
    adata = augment_gene_data(
        adata,
        adjacent_weight=adjacent_weight
    )

    return adata


# ==================== 3. 数据处理类 (从PlantST整合的关键方法) ====================

class DataProcessor:
    """
    数据处理类 - 整合自PlantST

    功能:
        - 加载不同平台的空间转录组数据
        - 对基因表达进行增强
        - 预处理数据用于下游分析
    """

    def __init__(self, save_path="./", use_gpu=True, seed=42):
        """
        初始化数据处理器

        参数:
            save_path: 保存路径
            use_gpu: 是否使用GPU
            seed: 随机种子
        """
        self.save_path = save_path
        self.use_gpu = use_gpu
        self.seed = seed

    def get_adata(self, platform, data_path, data_name, verbose=True):
        """
        加载不同平台的空间转录组数据

        参数:
            platform: 平台类型 ('Visium', 'ST', 'MERFISH', 'slideSeq', 'stereoSeq')
            data_path: 数据文件夹路径
            data_name: 数据文件名
            verbose: 是否保存原始数据

        返回:
            adata: AnnData对象
        """
        assert platform in ['Visium', 'ST', 'MERFISH', 'slideSeq', 'stereoSeq'], \
            f"不支持的平台: {platform}"

        if platform == 'Visium':
            adata = read_10X_Visium(os.path.join(data_path, data_name))
        elif platform == 'ST':
            adata = ReadOldST(os.path.join(data_path, data_name))
        elif platform == 'MERFISH':
            adata = read_merfish(os.path.join(data_path, data_name))
        elif platform == 'slideSeq':
            adata = read_SlideSeq(os.path.join(data_path, data_name))
        elif platform == 'stereoSeq':
            adata = read_stereoSeq(os.path.join(data_path, data_name))
        else:
            raise ValueError(f"{platform} does not support.")

        if verbose:
            save_data_path = Path(os.path.join(self.save_path, "Data", data_name))
            save_data_path.mkdir(parents=True, exist_ok=True)
            adata.write(os.path.join(save_data_path, f'{data_name}_raw.h5ad'), compression="gzip")

        return adata

    def get_augment(self, adata, spatial_type, adjacent_weight=0.4, neighbour_k=8,
                    spatial_k=20, n_components=100, gb_dist_type="correlation",
                    use_data="raw"):
        """
        参数:
            adata: AnnData对象
            spatial_type: 空间算法类型
            adjacent_weight: 邻近点在增强过程中的占比
            neighbour_k: 每个spot的邻域中最近邻的点数量
            spatial_k: 基于空间位置的邻域范围
            n_components: PCA组件数量
            gb_dist_type: 基因表达距离的度量方法
            use_data: 使用的数据类型

        返回:
            adata: 增强后的AnnData对象
        """
        adata = augment_adata_no_morphology(
            adata,
            gb_dist_type=gb_dist_type,
            n_components=n_components,
            use_data=use_data,
            neighbour_k=neighbour_k,
            adjacent_weight=adjacent_weight,
            spatial_k=spatial_k,
            spatial_type=spatial_type,
        )
        print("Step 1: Augment molecule expression is Done!")
        return adata

    def data_preprocess_ccc(self, adata):
        """
        预处理数据用于细胞通信分析

        参数:
            adata: 包含增强基因表达的AnnData对象

        返回:
            enhanced_expr: 处理后的增强表达矩阵
        """
        augment_X = adata.obsm["augment_gene_data"].astype(np.float64)
        ad_tmp = sc.AnnData(augment_X)
        sc.pp.normalize_total(ad_tmp, target_sum=1)
        sc.pp.log1p(ad_tmp)
        enhanced_expr = ad_tmp.X.copy()
        if sparse.issparse(enhanced_expr):
            enhanced_expr = enhanced_expr.toarray()
        return enhanced_expr

    def data_preprocess_identify(self, adata, pca_n_comps=200):
        """
        预处理数据用于空间域识别

        参数:
            adata: AnnData对象
            pca_n_comps: PCA组件数量

        返回:
            pca_features: PCA降维后的特征矩阵
        """
        augment_X = adata.obsm["augment_gene_data"].astype(np.float64)
        ad_tmp = sc.AnnData(augment_X)
        sc.pp.normalize_total(ad_tmp, target_sum=1)
        sc.pp.log1p(ad_tmp)
        sc.pp.scale(ad_tmp)
        sc.pp.pca(ad_tmp, n_comps=pca_n_comps)
        adata.obsm["augment_gene_data_pca"] = ad_tmp.obsm["X_pca"].astype(np.float32)
        return adata.obsm["augment_gene_data_pca"]


# ==================== 4. PlantCCC 预处理器类 ====================

class PreprocessorPlus:
    """
    PlantCCC 预处理器

    功能：
        - 接收外部增强的表达矩阵
        - 构建细胞通信图
        - 边属性：[距离权重, 通信分数, 关系ID]

    输入：
        - adata: 原始 AnnData 对象（包含空间坐标）
        - enhanced_expression: 增强的基因表达矩阵（N × D）

    输出：
        - 图文件
        - 元数据文件（barcode、坐标、聚类信息）
    """

    def __init__(self, data_name, adata, enhanced_expression, **kwargs):
        """
        初始化预处理器

        参数：
            data_name: 数据集名称
            adata: scanpy AnnData 对象
            enhanced_expression: 增强的基因表达矩阵（N × D）
            **kwargs: 其他可选参数
        """
        # ========== 必选参数 ==========
        self.data_name = data_name
        self.adata = adata
        self.enhanced_expression = enhanced_expression

        # ========== 输出路径 ==========
        self.data_to = kwargs.get('data_to', 'input_graph/')
        self.metadata_to = kwargs.get('metadata_to', 'metadata/')

        # ========== 聚类文件路径（必需） ==========
        self.cluster_h5ad_path = kwargs.get('cluster_h5ad_path', None)

        # ========== 距离参数 ==========
        self.base_distance_multiplier = kwargs.get('base_distance_multiplier', 1.5)

        # ========== 配体-受体数据库 ==========
        self.database_path = kwargs.get('database_path', 'database/poplar_lr_pairs.csv')

        # ========== 表达阈值（百分位数） ==========
        self.threshold_gene_exp = kwargs.get('threshold_gene_exp', 98)

        # ========== 自分泌控制 ==========
        self.block_autocrine = kwargs.get('block_autocrine', 1)

        # ========== 内部变量（运行时填充） ==========
        self.coordinates = None
        self.cell_barcode = None
        self.cell_vs_gene = None
        self.gene_ids = None
        self.ligand_dict_dataset = None
        self.l_r_pair = None
        self.cell_percentile = None
        self.cluster_labels = None
        self.weightdict = None

        # 创建输出目录
        self._create_output_directories()

    # ==================== 输入输出 ====================

    def _create_output_directories(self):
        """创建输出目录"""
        if self.data_to.rstrip('/').endswith('input_graph'):
            self.data_to = os.path.join(self.data_to, self.data_name)
        os.makedirs(self.data_to, exist_ok=True)

        if self.metadata_to.rstrip('/').endswith('metadata'):
            self.metadata_to = os.path.join(self.metadata_to, self.data_name)
        os.makedirs(self.metadata_to, exist_ok=True)

    def _extract_from_adata(self):
        """从 adata 提取坐标/条形码/基因名；用增强矩阵替换表达值（保持 adata 的行列顺序）"""
        print('正在从 adata 和增强表达矩阵提取信息...')

        # 1) 坐标（优先 'spatial'，否则 'X_spatial'）
        if 'spatial' in self.adata.obsm:
            self.coordinates = np.asarray(self.adata.obsm['spatial'], dtype=np.float32)
        elif 'X_spatial' in self.adata.obsm:
            self.coordinates = np.asarray(self.adata.obsm['X_spatial'], dtype=np.float32)
        else:
            raise ValueError("未找到空间坐标: adata.obsm['spatial'] 或 adata.obsm['X_spatial']")

        # 2) 条形码 & 基因名（完全沿用 adata）
        self.cell_barcode = np.asarray(self.adata.obs_names, dtype=str)
        self.gene_ids = np.asarray(self.adata.var_names, dtype=str)

        n_cells = self.cell_barcode.size
        n_genes = self.gene_ids.size

        # 3) 表达矩阵：首选你传入的增强矩阵（ndarray），否则退回 adata.X/raw/layers['counts']
        if self.enhanced_expression is not None:
            X = self.enhanced_expression
        else:
            if hasattr(self.adata, "raw") and (self.adata.raw is not None):
                X = self.adata.raw.X
            elif hasattr(self.adata, "layers") and ('counts' in self.adata.layers):
                X = self.adata.layers['counts']
            else:
                X = self.adata.X

        # 稀疏转稠密、统一 dtype
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        # 一致性检查：行=细胞数，列=基因数（严格对齐 adata 的顺序）
        if X.shape != (n_cells, n_genes):
            raise ValueError(
                f"增强矩阵形状为 {X.shape}，但期望 ({n_cells}, {n_genes})，"
                "请确认 data_preprocess_ccc(adata) 输出与 adata.obs_names/var_names 对齐。"
            )

        self.cell_vs_gene = X

        print(f'✅ 细胞数量: {n_cells}')
        print(f'✅ 表达矩阵形状: {self.cell_vs_gene.shape}')
        print(f'✅ 基因数量: {n_genes}')
        print(f'✅ 空间坐标形状: {self.coordinates.shape}')

    # ==================== 聚类加载 ====================

    def _load_clusters(self):
        """加载聚类标签"""
        if not os.path.exists(self.cluster_h5ad_path):
            raise FileNotFoundError(
                f"未找到聚类文件: {self.cluster_h5ad_path}\n"
                f"聚类文件是必需的"
            )

        ad = sc.read_h5ad(self.cluster_h5ad_path)

        # 查找聚类列（优先级顺序）
        key = None
        for cand in ['cluster', 'clusters', 'leiden', 'louvain']:
            if cand in ad.obs.columns:
                key = cand
                break

        # 如果没有标准列名，查找符合条件的列
        if key is None:
            for c in ad.obs.columns:
                s = ad.obs[c]
                if (pd.api.types.is_integer_dtype(s) or pd.api.types.is_categorical_dtype(
                        s)) and 2 <= s.nunique() <= 200:
                    key = c
                    break

        if key is None:
            raise ValueError(
                f"在 {self.cluster_h5ad_path} 中未找到合适的聚类列\n"
                f"预期列名: 'cluster', 'clusters', 'leiden', 或 'louvain'"
            )

        # 对齐 barcode
        obs = ad.obs
        m = {bc: i for i, bc in enumerate(obs.index.astype(str))}
        labels = np.full(self.cell_barcode.shape[0], -1, dtype=int)
        series = obs[key]

        # 转换分类变量为整数
        if pd.api.types.is_categorical_dtype(series):
            series = series.cat.codes

        # 匹配 barcode
        for i, bc in enumerate(self.cell_barcode.astype(str)):
            j = m.get(bc, None)
            if j is not None:
                labels[i] = int(series.iloc[j])

        n_unmatched = np.sum(labels == -1)
        if n_unmatched > 0:
            print(f"⚠️ 警告: {n_unmatched} 个细胞未在聚类文件中找到")

        self.cluster_labels = labels
        print(f"✅ 已加载聚类标签: {self.cluster_h5ad_path}")
        print(f"   聚类列: '{key}'")
        print(f"   聚类数量: {len(np.unique(labels[labels >= 0]))}")

    # ==================== 邻域构建 ====================

    def _build_neighbors(self):
        """构建尺度邻域（基于半径查询）"""
        print('正在构建邻域...')

        # 计算最近邻距离中位数
        nbrs_temp = NearestNeighbors(n_neighbors=7, algorithm='kd_tree', n_jobs=-1)
        nbrs_temp.fit(self.coordinates)
        temp_dists, _ = nbrs_temp.kneighbors(self.coordinates)
        median_nn_dist = np.median(temp_dists[:, 1])

        # 计算距离阈值
        max_distance = self.base_distance_multiplier * median_nn_dist

        print(f'  最近邻距离中位数: {median_nn_dist:.2f}')
        print(f'  距离阈值: {max_distance:.2f} (= {self.base_distance_multiplier:.2f} × {median_nn_dist:.2f})')

        # 使用半径查询
        nbrs = NearestNeighbors(radius=max_distance, algorithm='kd_tree', n_jobs=-1)
        nbrs.fit(self.coordinates)
        distances, indices = nbrs.radius_neighbors(self.coordinates, return_distance=True)

        weightdict = defaultdict(dict)

        for i in range(len(indices)):
            idx_row = np.array(indices[i])
            dist_row = np.array(distances[i])

            # 排除自身
            keep = idx_row != i
            idx_row = idx_row[keep]
            dist_row = dist_row[keep]

            if len(dist_row) == 0:
                continue

            # 权重归一化
            dmax, dmin = float(dist_row.max()), float(dist_row.min())
            denom = (dmax - dmin + 1e-9)
            for jj, d in zip(idx_row, dist_row):
                w = 1.0 - (float(d) - dmin) / denom
                weightdict[i][int(jj)] = float(w)

        total_neighbors = sum(len(v) for v in weightdict.values())
        avg_neighbors = total_neighbors / len(weightdict) if weightdict else 0

        print(f'  平均邻居数: {avg_neighbors:.1f}')
        print(f'  总边数: {total_neighbors}')

        self.weightdict = weightdict

    # ==================== 配体-受体数据库 ====================

    def _read_ligand_receptor_database(self):
        """读取配体-受体数据库"""
        print('正在加载配体-受体数据库...')
        df = pd.read_csv(self.database_path, sep=",")
        print(f'✅ 数据库已加载: {len(df)} 个 L-R 对')

        # 筛选数据集中存在的基因
        gene_info = {g: '' for g in self.gene_ids}
        self.ligand_dict_dataset = defaultdict(list)

        for i in range(df.shape[0]):
            L = df["Ligand"][i]
            R = df["Receptor"][i]
            if L in gene_info and R in gene_info:
                self.ligand_dict_dataset[L].append(R)
                gene_info[L] = 'included'
                gene_info[R] = 'included'

        print(f'✅ 配体数量: {len(self.ligand_dict_dataset.keys())}')
        included = [g for g in gene_info if gene_info[g] == 'included']
        print(f'✅ L/R 基因在数据集中: {len(included)} / {len(self.gene_ids)}')

        # 为每个 L-R 对分配唯一 ID
        self.l_r_pair = dict()
        lr_id = 0
        for L in list(self.ligand_dict_dataset.keys()):
            self.ligand_dict_dataset[L] = list(set(self.ligand_dict_dataset[L]))
            self.l_r_pair[L] = dict()
            for R in self.ligand_dict_dataset[L]:
                self.l_r_pair[L][R] = lr_id
                lr_id += 1
        print(f'✅ L-R 对总数: {lr_id}')

        # 保存配体列表和 L-R 对映射
        with gzip.open(os.path.join(self.metadata_to, "ligand_list.pkl"), 'wb') as f:
            pickle.dump(list(self.ligand_dict_dataset.keys()), f)
        with gzip.open(os.path.join(self.metadata_to, "l_r_pair.pkl"), 'wb') as f:
            pickle.dump(self.l_r_pair, f)

    def _calculate_gene_expression_thresholds(self):
        """计算基因表达阈值（基于百分位，矢量化实现）"""
        print('正在计算基因表达阈值...')
        # 若是稀疏，先转稠密（更快；内存允许时建议这样做）
        X = self.cell_vs_gene.toarray() if sp.issparse(self.cell_vs_gene) else np.asarray(self.cell_vs_gene)

        # 一次性按行求百分位
        self.cell_percentile = np.percentile(X, self.threshold_gene_exp, axis=1).astype(np.float32)

        # 避免阈值==行最小值的退化情况：改用行最大值
        row_min = X.min(axis=1)
        row_max = X.max(axis=1)
        self.cell_percentile = np.where(self.cell_percentile == row_min, row_max, self.cell_percentile).astype(
            np.float32)

        print(f'✅ 阈值计算完成（百分位数={self.threshold_gene_exp}）')

    # ==================== 图构建 ====================

    def _build_interaction_network(self):
        """构建尺度细胞通信网络"""
        print('正在构建细胞通信网络...')

        ligand_list = list(self.ligand_dict_dataset.keys())
        gene_index = {g: i for i, g in enumerate(self.gene_ids)}

        row_col, edge_weight, lig_rec = [], [], []

        with tqdm(total=len(ligand_list),
                  desc="处理配体",
                  bar_format="{l_bar}{bar} [ {n_fmt}/{total_fmt} ]") as pbar:

            for g, L in enumerate(ligand_list):
                if L not in gene_index:
                    pbar.update(1)
                    continue

                li = gene_index[L]

                for i in range(self.cell_vs_gene.shape[0]):
                    # 检查配体表达
                    if self.cell_vs_gene[i][li] < self.cell_percentile[i]:
                        continue

                    # 遍历邻居
                    for j in self.weightdict[i].keys():
                        # 是否禁止自分泌
                        if (self.block_autocrine == 1) and (i == j):
                            continue

                        # 遍历对应的受体
                        for R in self.ligand_dict_dataset[L]:
                            if R not in gene_index:
                                continue
                            rj = gene_index[R]

                            # 检查受体表达
                            if self.cell_vs_gene[j][rj] < self.cell_percentile[j]:
                                continue

                            # 计算通信分数
                            comm = float(self.cell_vs_gene[i][li] * self.cell_vs_gene[j][rj])
                            if comm <= 0:
                                continue

                            # 获取关系ID
                            rel_id = self.l_r_pair[L][R]

                            # 添加边
                            row_col.append([i, j])
                            edge_weight.append([
                                self.weightdict[i][j],  # 距离权重
                                comm,  # 通信分数
                                rel_id  # 关系ID
                            ])
                            lig_rec.append([L, R])

                pbar.update(1)

        self._pack = (row_col, edge_weight, lig_rec)
        print(f'\n✅ 图构建完成: {len(row_col)} 条边')

    # ==================== 结果保存 ====================

    def _save_graph(self):
        """保存图文件"""
        N = self.cell_vs_gene.shape[0]
        out_path = os.path.join(self.data_to, f'{self.data_name}_adjacency_records')

        row_col, edge_weight, lig_rec = self._pack

        with gzip.open(out_path, 'wb') as fp:
            pickle.dump([row_col, edge_weight, lig_rec, N], fp)

        print(f'\n✅ 图已保存: {out_path}')
        print(f'   节点数: {N}')
        print(f'   边数: {len(row_col)}')

    def _save_metadata(self):
        """保存元数据"""
        if self.cluster_labels is None:
            self.cluster_labels = -np.ones(self.cell_barcode.shape[0], dtype=int)

        # 保存barcode信息
        barcode_info = [
            [str(self.cell_barcode[i]),
             float(self.coordinates[i, 0]),
             float(self.coordinates[i, 1]),
             int(self.cluster_labels[i])]
            for i in range(self.cell_barcode.shape[0])
        ]
        barcode_path = os.path.join(self.metadata_to, f'{self.data_name}_barcode_info')
        with gzip.open(barcode_path, 'wb') as fp:
            pickle.dump(barcode_info, fp)

        print(f'✅ Barcode信息已保存: {barcode_path}')

    def _save_lr_statistics(self):
        """保存L-R对统计"""
        print('\n正在统计L-R对...')

        lr_count = defaultdict(int)
        for lr in self._pack[2]:
            lr_count[(lr[0], lr[1])] += 1

        lr_df = pd.DataFrame([
            {'Ligand': lr[0], 'Receptor': lr[1], 'Edge_Count': count}
            for lr, count in lr_count.items()
        ])
        lr_df = lr_df.sort_values('Edge_Count', ascending=False).reset_index(drop=True)

        lr_stats_path = os.path.join(self.metadata_to, f'{self.data_name}_LR_pairs_in_edges.csv')
        lr_df.to_csv(lr_stats_path, index=False)

        total_edges = len(self._pack[0])
        unique_lr_pairs = len(lr_count)
        database_lr_pairs = sum(len(self.ligand_dict_dataset[L]) for L in self.ligand_dict_dataset)
        coverage = unique_lr_pairs / database_lr_pairs * 100 if database_lr_pairs > 0 else 0

        print(f'\n✅ L-R对统计已保存: {lr_stats_path}')
        print(f'\n【边统计】')
        print(f'  - 总边数: {total_edges:,}')
        print(f'\n【L-R对统计】')
        print(f'  - 数据库总L-R对数: {database_lr_pairs}')
        print(f'  - 实际使用的L-R对数: {unique_lr_pairs}')
        print(f'  - L-R对使用率: {coverage:.2f}%')

    # ==================== 主流程 ====================

    def run(self):
        """运行单尺度预处理"""
        print('=' * 60)
        print('开始 PlantCCC 预处理')
        print('=' * 60)

        self._extract_from_adata()
        self._load_clusters()
        self._build_neighbors()
        self._read_ligand_receptor_database()
        self._calculate_gene_expression_thresholds()
        self._build_interaction_network()
        self._save_graph()
        self._save_metadata()
        self._save_lr_statistics()

        print('\n' + '=' * 60)
        print('✅ 预处理完成！')
        print('=' * 60)