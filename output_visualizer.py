
import numpy as np
import pickle
import matplotlib

matplotlib.use('Agg')
from scipy.sparse import csr_matrix
from collections import defaultdict
import pandas as pd
import gzip
import os
from scipy.sparse.csgraph import connected_components
from pyvis.network import Network
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import altair as alt
import altairThemes  # 假设该模块存在于当前目录或系统路径中
import gc
import copy

# # 注册并启用Altair主题
# alt.themes.register("publishTheme", altairThemes.publishTheme)
# alt.themes.enable("publishTheme")


class PlantCCCVisualizer:
    """
    PlantCCC可视化工具类

    新增：
    - 在导出的CCC列表CSV中包含 from_barcode/to_barcode 与 from_cluster/to_cluster 四列
      （cluster 即注释文件Type列；若无显式domain/cluster列，则回落为Type）
    - 网络图边的提示文本包含 clusterA→clusterB
    - 直方图阶段额外导出按 cluster 对统计的 LR 频次表
    """

    def __init__(self, data_name, model_name, **kwargs):
        """
        初始化PlantCCC可视化工具

        参数:
            data_name (str): 数据集名称
            model_name (str): 训练模型名称
            **kwargs: 其他可选参数
                top_edge_count (int): 要绘制的顶级通信数量，默认为1500
                metadata_from (str): 元数据路径，默认为'metadata/'
                output_path (str): 结果保存路径，默认为'output/'
                top_ccc_file (str): 顶级CCC文件路径，默认为空
                output_name (str): 输出文件前缀，默认为空
                filter (int): 是否过滤CCC，默认为0
                filter_by_ligand_receptor (str): 按配体-受体对过滤，默认为空
                filter_by_annotation (str): 按细胞或斑点类型过滤，默认为空
                filter_by_component (int): 按组件ID过滤，默认为-1
                sort_by_attentionScore (int): 是否按注意力分数排序直方图，默认为-1
                min_distance (float): 最小距离过滤（可选）
        """
        # 基础参数
        self.data_name = data_name
        self.model_name = model_name

        # 可选参数，设置默认值
        self.top_edge_count = kwargs.get('top_edge_count', 1500)
        self.metadata_from = kwargs.get('metadata_from', 'metadata/')
        self.output_path = kwargs.get('output_path', 'output/')
        self.barcode_info_file = kwargs.get('barcode_info_file', '')
        self.annotation_file_path = kwargs.get('annotation_file_path', '')
        self.selfloop_info_file = kwargs.get('selfloop_info_file', '')
        self.top_ccc_file = kwargs.get('top_ccc_file', '')
        self.output_name = kwargs.get('output_name', '')
        self.filter = kwargs.get('filter', 0)
        self.filter_by_ligand_receptor = kwargs.get('filter_by_ligand_receptor', '')
        self.filter_by_annotation = kwargs.get('filter_by_annotation', '')
        self.filter_by_component = kwargs.get('filter_by_component', -1)
        self.sort_by_attentionScore = kwargs.get('sort_by_attentionScore', -1)
        self.min_distance = kwargs.get('min_distance', None)

        # 数据存储变量
        self.barcode_info = None
        self.self_loop_found = None
        self.barcode_type = None            # barcode -> Type (此处当作 cluster)
        self.barcode_domain_map = None      # barcode -> domain/cluster（优先domain列，缺省回落Type）
        self.annotation_df = None
        self.df = None
        self.csv_record_final = None
        self.active_spot = None

        # 处理路径
        self._process_paths()

        print(f"PlantCCCVisualizer 初始化完成，数据集: {self.data_name}, 模型: {self.model_name}")
        print(f"将绘制前 {self.top_edge_count} 个通信。如需更改，请调整 top_edge_count 参数")

    def _process_paths(self):
        """处理输入输出路径"""
        if self.metadata_from == 'metadata/':
            self.metadata_from = os.path.join(self.metadata_from, self.data_name) + '/'
        if self.output_path == 'output/':
            self.output_path = os.path.join(self.output_path, self.data_name) + '/'

        os.makedirs(self.output_path, exist_ok=True)

        if not self.output_name:
            self.output_name = os.path.join(self.output_path, self.model_name)
        else:
            self.output_name = os.path.join(self.output_path, self.output_name)

    def load_barcode_info(self):
        """加载barcode信息数据"""
        print("加载barcode信息...")
        if not self.barcode_info_file:
            file_path = os.path.join(self.metadata_from, f'{self.data_name}_barcode_info')
        else:
            file_path = self.barcode_info_file

        with gzip.open(file_path, 'rb') as fp:
            self.barcode_info = pickle.load(fp)
        print(f"✅ 加载barcode信息，共 {len(self.barcode_info)} 条记录")

    def load_selfloop_info(self):
        """加载自环信息数据"""
        print("加载自环信息...")
        if not self.selfloop_info_file:
            file_path = os.path.join(self.metadata_from, f'{self.data_name}_self_loop_record')
        else:
            file_path = self.selfloop_info_file

        if not os.path.exists(file_path):
            print("⚠️ 未找到自环信息文件，跳过")
            self.self_loop_found = {}
            return

        with gzip.open(file_path, 'rb') as fp:
            self.self_loop_found = pickle.load(fp)
        print("✅ 自环信息加载完成")

    def load_annotations(self):
        """加载注释信息（条形码→类型/域/cluster）"""
        print("加载注释信息...")
        self.barcode_type = dict()
        self.barcode_domain_map = dict()
        self.annotation_df = None

        if not self.annotation_file_path:
            for i in range(len(self.barcode_info)):
                self.barcode_type[self.barcode_info[i][0]] = ''
                self.barcode_domain_map[self.barcode_info[i][0]] = ''
            print("⚠️ 未提供注释文件，使用空注释信息")
            return

        df = pd.read_csv(self.annotation_file_path)
        self.annotation_df = df.copy()

        # 识别条形码列
        barcode_col = None
        for c in ["Barcode", "barcode", "barcodes", "spot_id", "spotID", "cell_id"]:
            if c in df.columns:
                barcode_col = c
                break
        if barcode_col is None:
            raise ValueError("注释文件中找不到条形码列")

        # 识别类型列（此处被当作 cluster）
        type_col = None
        for c in ["Type", "type", "cell_type", "cellType", "annotation"]:
            if c in df.columns:
                type_col = c
                break

        # 识别域列（cluster/layer/domain 等）
        domain_col = None
        for c in ["cluster", "Cluster", "layer", "Layer", "domain", "Domain"]:
            if c in df.columns:
                domain_col = c
                break

        # 建立映射
        for _, r in df.iterrows():
            b = str(r[barcode_col])
            if type_col is not None:
                self.barcode_type[b] = r[type_col]
            else:
                self.barcode_type[b] = ''
            if domain_col is not None:
                self.barcode_domain_map[b] = str(r[domain_col])
            else:
                # 若缺失 domain 类列，则用 Type 兜底
                self.barcode_domain_map[b] = str(self.barcode_type[b])

        print(
            f"✅ 加载注释信息：type={type_col or '无'}, domain={domain_col or type_col or '无'}；共 {len(self.barcode_type)} 条记录")

    def _guess_ccc_file(self):
        """自动猜测 CCC文件路径"""
        if self.top_ccc_file:
            return self.top_ccc_file
    def _normalize_ccc_df(self):
        """
        统一 CSV列结构：
        [from_cell, to_cell, ligand, receptor, rank, component, from_id, to_id, score, distance]
        并新增：from_barcode, to_barcode, from_cluster, to_cluster
        """
        df = self.df.copy()

        # from/to 的 id 映射
        barcode2idx = {str(self.barcode_info[i][0]): i for i in range(len(self.barcode_info))}

        def _to_idx(x):
            sx = str(x)
            if sx in barcode2idx:
                return barcode2idx[sx]
            try:
                return int(float(sx))
            except Exception:
                return np.nan

        if 'from_id' in df.columns and 'to_id' in df.columns:
            pass
        elif 'from_cell' in df.columns and 'to_cell' in df.columns:
            df['from_id'] = df['from_cell'].apply(_to_idx).astype('Int64')
            df['to_id'] = df['to_cell'].apply(_to_idx).astype('Int64')
            before = len(df)
            df = df.dropna(subset=['from_id', 'to_id']).copy()
            df['from_id'] = df['from_id'].astype(int)
            df['to_id'] = df['to_id'].astype(int)
            if len(df) < before:
                print(f"⚠️ 映射失败并丢弃 {before - len(df)} 行")
        else:
            raise ValueError("找不到 from_id/to_id 或 from_cell/to_cell 列")

        # 分数列统一为 'score'
        score_col = None
        for c in ['score', 'norm_score', 'raw_score', 'attention_score', 'edge_score']:
            if c in df.columns:
                score_col = c
                break
        if score_col is None:
            df['score'] = 1.0
        elif score_col != 'score':
            df = df.rename(columns={score_col: 'score'})
        df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0.0)

        # rank
        if 'rank' not in df.columns:
            df['rank'] = df['score'].rank(method='first', ascending=False).astype(int)

        # 基本缺省列
        for c in ['from_cell', 'to_cell', 'ligand', 'receptor']:
            if c not in df.columns:
                df[c] = ''
        if 'component' not in df.columns:
            df['component'] = 0

        # 距离列（基于 from_id/to_id 与 barcode_info 中的坐标）
        def _dist(row):
            i, j = int(row['from_id']), int(row['to_id'])
            xi, yi = float(self.barcode_info[i][1]), float(self.barcode_info[i][2])
            xj, yj = float(self.barcode_info[j][1]), float(self.barcode_info[j][2])
            return np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

        df['distance'] = df.apply(_dist, axis=1)

        # --- 新增：把条形码 & cluster（Type）也写进 df ---
        id_to_barcode = {i: str(self.barcode_info[i][0]) for i in range(len(self.barcode_info))}
        df['from_barcode'] = df['from_id'].map(lambda i: id_to_barcode.get(int(i), '') if pd.notna(i) else '')
        df['to_barcode']   = df['to_id'].map(lambda i: id_to_barcode.get(int(i), '') if pd.notna(i) else '')
        df['from_cluster'] = df['from_barcode'].map(lambda b: self.barcode_type.get(b, ''))
        df['to_cluster']   = df['to_barcode'].map(lambda b: self.barcode_type.get(b, ''))

        # 按rank排序
        df = df.sort_values(['rank', 'score'], ascending=[True, False]).reset_index(drop=True)

        # 可选：按最小距离过滤
        if self.min_distance is not None:
            before = len(df)
            df = df[df['distance'] >= float(self.min_distance)].copy()
            print(f"📏 按 min_distance 过滤：{before} -> {len(df)}")

        # 列顺序（在原先基础上插入新增列）
        base_cols = [
            'from_cell', 'to_cell',
            'from_barcode', 'to_barcode',
            'from_cluster', 'to_cluster',
            'ligand', 'receptor', 'rank', 'component', 'from_id', 'to_id', 'score'
        ]
        # 确保存在的列都被包含
        base_cols = [c for c in base_cols if c in df.columns]
        extra_cols = [c for c in df.columns if c not in base_cols]
        ordered = base_cols + extra_cols

        # 设定索引（基于列名定位，避免新增列打乱）
        self.from_id_idx = ordered.index('from_id')
        self.to_id_idx = ordered.index('to_id')
        self.score_idx = ordered.index('score')
        self.component_idx = ordered.index('component')

        return df[ordered], ordered

    def load_ccc_data(self):
        """加载 CCC数据"""
        print("加载 CCC数据...")
        self.in_file = self._guess_ccc_file()
        print(f"  读取: {self.in_file}")
        self.df = pd.read_csv(self.in_file)
        print(f"✅ 加载CCC数据，共 {len(self.df)} 条记录")

    def preprocess_ccc_data(self):
        """预处理 CCC数据"""
        print("预处理 CCC数据...")
        df_norm, columns = self._normalize_ccc_df()

        # 只保留前 top_edge_count
        if self.top_edge_count != -1:
            df_norm = df_norm.iloc[:min(self.top_edge_count, len(df_norm))].copy()

        records = df_norm.values.tolist()

        # dummy 行
        dummy = [''] * len(columns)
        dummy[self.component_idx] = 0
        dummy[self.from_id_idx] = 0
        dummy[self.to_id_idx] = 0
        dummy[self.score_idx] = 0

        self.csv_record_final = [columns] + records + [dummy]

        print(f"  列: {columns[:10]}{' ...' if len(columns) > 10 else ''}")
        print(
            f"  索引: from_id={self.from_id_idx}, to_id={self.to_id_idx}, component={self.component_idx}, score={self.score_idx}")
        print(f"✅ 预处理完成，保留 {len(records)} 条有效记录\n")

    def find_connected_components(self):
        """分析并找到连接组件"""
        print("寻找连接组件...")
        connecting_edges = np.zeros((len(self.barcode_info), len(self.barcode_info)))

        for k in range(1, len(self.csv_record_final) - 1):
            i = self.csv_record_final[k][self.from_id_idx]
            j = self.csv_record_final[k][self.to_id_idx]
            connecting_edges[i][j] = 1

        graph = csr_matrix(connecting_edges)
        n_components, labels = connected_components(
            csgraph=graph, directed=True, connection='weak', return_labels=True
        )
        print(f"找到 {n_components} 个连接组件")

        count_points_component = np.zeros(n_components)
        for i in range(len(labels)):
            count_points_component[labels[i]] += 1

        id_label = 2
        index_dict = dict()
        for i in range(count_points_component.shape[0]):
            if count_points_component[i] > 1:
                index_dict[i] = id_label
                id_label += 1

        print(f"唯一组件数量: {id_label}")

        for i in range(len(self.barcode_info)):
            if count_points_component[labels[i]] > 1:
                self.barcode_info[i][3] = index_dict[labels[i]]
            elif connecting_edges[i][i] == 1 and (i in self.self_loop_found and i in self.self_loop_found[i]):
                self.barcode_info[i][3] = 1
            else:
                self.barcode_info[i][3] = 0

        for record in range(1, len(self.csv_record_final) - 1):
            i = self.csv_record_final[record][self.from_id_idx]
            label = self.barcode_info[i][3]
            self.csv_record_final[record][self.component_idx] = label

        self.id_label = id_label

    def _normalize_lr_filters(self):
        """
        规范化 self.filter_by_ligand_receptor，返回一个 {(ligand, receptor), ...} 的集合。
        支持：
          - "L-R" 字符串
          - ["L-R", "A-B"] 列表/元组/集合
          - [("L","R"), ("A","B")] 形式
        """
        v = self.filter_by_ligand_receptor
        pairs = set()
        if v is None or v == '' or v == []:
            return pairs

        # 统一成列表
        if isinstance(v, str):
            items = [v]
        elif isinstance(v, (list, tuple, set)):
            items = list(v)
        else:
            items = [str(v)]

        for it in items:
            if isinstance(it, (list, tuple)) and len(it) == 2:
                l, r = it[0], it[1]
            else:
                s = str(it).strip()
                # 兼容 "L->R" 写法
                s = s.replace('->', '-')
                parts = s.split('-', 1)
                if len(parts) != 2:
                    continue
                l, r = parts[0], parts[1]
            pairs.add((str(l).strip(), str(r).strip()))
        return pairs

    def filter_ccc_data(self):
        """根据条件过滤CCC数据（支持多种 L-R 过滤；按列名解析以适配新增列）"""
        if self.filter != 1:
            return

        print("过滤CCC数据...")
        header = self.csv_record_final[0]

        # 这些索引用列名找，避免列顺序变化带来的问题
        lig_idx = header.index('ligand') if 'ligand' in header else 2
        rec_idx = header.index('receptor') if 'receptor' in header else 3
        comp_idx = self.component_idx  # 已在预处理阶段记录

        # annotation 相关列（有就用，没有就兜底）
        from_barcode_idx = header.index('from_barcode') if 'from_barcode' in header else None
        to_barcode_idx = header.index('to_barcode') if 'to_barcode' in header else None
        from_cluster_idx = header.index('from_cluster') if 'from_cluster' in header else None
        to_cluster_idx = header.index('to_cluster') if 'to_cluster' in header else None

        # 规范化 L-R 过滤集合
        lr_set = self._normalize_lr_filters()

        csv_record_final_temp = [header]
        component_dictionary_dummy = dict()

        for record_idx in range(1, len(self.csv_record_final) - 1):
            record = self.csv_record_final[record_idx]
            keep = False

            # 1) 按组件过滤
            if self.filter_by_component != -1:
                keep = (record[comp_idx] == int(self.filter_by_component))

            # 2) 按 L-R 过滤（支持多个 pair）
            elif lr_set:
                keep = ((record[lig_idx], record[rec_idx]) in lr_set)

            # 3) 按注释过滤（优先用 cluster 列；否则用 barcode 映射回 self.barcode_type）
            elif self.filter_by_annotation:
                left = right = None
                if from_cluster_idx is not None and to_cluster_idx is not None:
                    left = str(record[from_cluster_idx])
                    right = str(record[to_cluster_idx])
                elif from_barcode_idx is not None and to_barcode_idx is not None:
                    left = str(self.barcode_type.get(str(record[from_barcode_idx]), ''))
                    right = str(self.barcode_type.get(str(record[to_barcode_idx]), ''))
                keep = (left == self.filter_by_annotation and right == self.filter_by_annotation)

            # 4) 默认不过滤
            else:
                keep = True

            if keep:
                csv_record_final_temp.append(record)

            # 保留一个该组件的样例记录（延续你原本的逻辑）
            if record[comp_idx] not in component_dictionary_dummy:
                component_dictionary_dummy[record[comp_idx]] = record

        # 把样例记录附加进去（若未被选中过）
        for component_id, rec in component_dictionary_dummy.items():
            if rec not in csv_record_final_temp:
                csv_record_final_temp.append(rec)

        csv_record_final_temp.append(self.csv_record_final[-1])
        self.csv_record_final = copy.deepcopy(csv_record_final_temp)
        print(f"✅ 过滤完成，保留 {len(self.csv_record_final) - 2} 条有效记录")

        # 重新同步 barcode_info 的 component 标记
        component_list = dict()
        for record_idx in range(1, len(self.csv_record_final) - 1):
            record = self.csv_record_final[record_idx]
            i = record[self.from_id_idx]
            j = record[self.to_id_idx]
            component_label = record[self.component_idx]
            self.barcode_info[i][3] = component_label
            self.barcode_info[j][3] = component_label
            component_list[component_label] = ''
        component_list[0] = ''
        self.unique_component_count = max(len(component_list.keys()), getattr(self, 'id_label', 0))
    def save_ccc_list(self):
        """保存CCC列表（首行即列名）"""
        print("保存CCC列表...")
        df = pd.DataFrame(self.csv_record_final)
        output_file = f"{self.output_name}_ccc_list_top{self.top_edge_count}.csv"
        df.to_csv(output_file, index=False, header=False)
        print(f"✅ CCC列表已保存至: {output_file}")

    def prepare_active_spots(self):
        """准备活跃点数据"""
        print("准备活跃点数据...")
        self.active_spot = defaultdict(list)

        for record_idx in range(1, len(self.csv_record_final) - 1):
            record = self.csv_record_final[record_idx]
            i = record[self.from_id_idx]
            j = record[self.to_id_idx]
            component_label = record[self.component_idx]
            opacity = np.float64(record[self.score_idx])

            pathology_label = self.barcode_type[self.barcode_info[i][0]]
            X, Y = self.barcode_info[i][1], -self.barcode_info[i][2]
            self.active_spot[i].append([pathology_label, component_label, X, Y, opacity])

            pathology_label = self.barcode_type[self.barcode_info[j][0]]
            X, Y = self.barcode_info[j][1], -self.barcode_info[j][2]
            self.active_spot[j].append([pathology_label, component_label, X, Y, opacity])

        opacity_list = []
        for i in self.active_spot:
            sum_opacity = [edges[4] for edges in self.active_spot[i]]
            avg_opacity = np.max(sum_opacity)
            opacity_list.append(avg_opacity)
            self.active_spot[i] = [
                self.active_spot[i][0][0], self.active_spot[i][0][1],
                self.active_spot[i][0][2], self.active_spot[i][0][3], avg_opacity
            ]

        self.min_opacity = np.min(opacity_list) if opacity_list else 0
        self.max_opacity = np.max(opacity_list) if opacity_list else 1

    def generate_component_plot(self):
        """生成组件散点图"""
        print("生成组件散点图...")
        data_list = {
            'pathology_label': [],
            'component_label': [],
            'X': [],
            'Y': [],
            'opacity': []
        }

        for i in range(len(self.barcode_info)):
            if i in self.active_spot:
                data_list['pathology_label'].append(self.active_spot[i][0])
                data_list['component_label'].append(self.active_spot[i][1])
                data_list['X'].append(self.active_spot[i][2])
                data_list['Y'].append(self.active_spot[i][3])
                opacity = (self.active_spot[i][4] - self.min_opacity) / (
                        self.max_opacity - self.min_opacity) if self.max_opacity > self.min_opacity else 0
                data_list['opacity'].append(opacity)
            else:
                data_list['pathology_label'].append(self.barcode_type[self.barcode_info[i][0]])
                data_list['component_label'].append(0)
                data_list['X'].append(self.barcode_info[i][1])
                data_list['Y'].append(-self.barcode_info[i][2])
                data_list['opacity'].append(0.1)

        data_list_pd = pd.DataFrame(data_list)
        id_label = len(set(data_list['component_label']))
        set1 = altairThemes.get_colour_scheme("Set1", id_label)
        set1[0] = '#000000'

        chart = alt.Chart(data_list_pd).mark_point(filled=True, opacity=1).encode(
            alt.X('X', scale=alt.Scale(zero=False)),
            alt.Y('Y', scale=alt.Scale(zero=False)),
            shape=alt.Shape('pathology_label:N'),
            color=alt.Color('component_label:N', scale=alt.Scale(range=set1)),
            tooltip=['component_label']
        )

        output_file = f"{self.output_name}_component_plot.html"
        chart.save(output_file)
        print(f"✅ 组件散点图已保存至: {output_file}")

    @staticmethod
    def preprocess_df(df):
        """预处理数据框"""
        df["ligand"] = df["ligand"].astype(str)
        df["receptor"] = df["receptor"].astype(str)
        df["ligand-receptor"] = df["ligand"] + '-' + df["receptor"]
        df["component"] = df["component"]
        return df

    @staticmethod
    def plot_histogram(df):
        # 统一为整数并确定 domain 顺序
        domain = sorted(df["component"].astype(int).unique().tolist())
        palette = altairThemes.get_colour_scheme("Set1", len(domain))
        if 0 in domain:
            palette[domain.index(0)] = "#000000"  # 只有 0 用黑色

        base = alt.Chart(df).mark_bar().encode(
            x=alt.X("ligand-receptor:N", axis=alt.Axis(labelAngle=45), sort='-y'),
            y=alt.Y("count()"),
            color=alt.Color("component:N", scale=alt.Scale(domain=domain, range=palette)),
            order=alt.Order("component:N", sort="ascending"),
            tooltip=["component"]
        )
        return base

    def generate_histograms(self):
        """生成直方图与导出统计表"""
        print("生成直方图...")
        # 直接用内存数据构造带列名的 DataFrame
        df_edges = pd.DataFrame(self.csv_record_final[1:-1], columns=self.csv_record_final[0])
        print(f"用于直方图生成的数据共 {len(df_edges)} 条记录")

        df_processed = self.preprocess_df(df_edges.copy())
        df_processed["component"] = df_processed["component"].astype(int)
        df_processed = df_processed[df_processed["component"] != 0]
        histogram = self.plot_histogram(df_processed)
        output_file = f"{self.output_name}_histogram_byFrequency_plot.html"
        histogram.save(output_file)
        print(f"✅ 频率直方图已保存至: {output_file}")

        # （1）整体 LR 频次表
        hist_count = defaultdict(list)
        for i in range(1, len(self.csv_record_final) - 1):
            lr_pair = f"{self.csv_record_final[i][self.csv_record_final[0].index('ligand')]}-" \
                      f"{self.csv_record_final[i][self.csv_record_final[0].index('receptor')]}"
            hist_count[lr_pair].append(1)

        lr_pair_count = []
        for lr_pair, counts in hist_count.items():
            lr_pair_count.append([lr_pair, np.sum(counts)])
        lr_pair_count = sorted(lr_pair_count, key=lambda x: x[1], reverse=True)

        data_list_pd = pd.DataFrame({
            'Ligand-Receptor Pairs': [item[0] for item in lr_pair_count],
            'Total Count': [item[1] for item in lr_pair_count]
        })
        output_file = f"{self.output_name}_histogram_byFrequency_table.csv"
        data_list_pd.to_csv(output_file, index=False)
        print(f"✅ 直方图数据表格已保存至: {output_file}")

        # （2）按 cluster 对细分：from_cluster→to_cluster × (ligand, receptor)
        needed = {'from_cluster', 'to_cluster', 'ligand', 'receptor'}
        if needed.issubset(df_edges.columns):
            df_edges['cluster_pair'] = df_edges['from_cluster'].astype(str) + '→' + df_edges['to_cluster'].astype(str)
            grp = (df_edges
                   .groupby(['cluster_pair', 'ligand', 'receptor'], dropna=False)
                   .size()
                   .reset_index(name='Count')
                   .sort_values('Count', ascending=False))

            out2 = f"{self.output_name}_histogram_byFrequency_byClusterPair.csv"
            grp.to_csv(out2, index=False)
            print(f"✅ 按 cluster 对统计的频次表已保存至: {out2}")

    def generate_attention_histogram(self):
        """按注意力分数生成直方图（可选）"""
        if self.sort_by_attentionScore != 1:
            return

        print("生成注意力分数直方图...")
        lr_score = defaultdict(list)
        for i in range(1, len(self.csv_record_final) - 1):
            ligand = self.csv_record_final[i][self.csv_record_final[0].index('ligand')]
            receptor = self.csv_record_final[i][self.csv_record_final[0].index('receptor')]
            lr_pair = f"{ligand}-{receptor}"
            lr_score[lr_pair].append(self.csv_record_final[i][self.score_idx])

        for key in lr_score:
            lr_score[key] = np.sum(lr_score[key])

        data_list_pd = pd.DataFrame({
            'Ligand-Receptor Pairs': list(lr_score.keys()),
            'Total Attention Score': list(lr_score.values())
        })

        chart = alt.Chart(data_list_pd).mark_bar().encode(
            x=alt.X("Ligand-Receptor Pairs:N", axis=alt.Axis(labelAngle=45), sort='-y'),
            y='Total Attention Score'
        )

        output_file = f"{self.output_name}_histogram_byAttention_plot.html"
        chart.save(output_file)
        print(f"✅ 注意力分数直方图已保存至: {output_file}")

    def generate_network_graph(self):
        """生成网络图（边提示包含 clusterA→clusterB）"""
        print("生成网络图...")

        # 收集组件标签
        component_list = defaultdict(str)
        for record_idx in range(1, len(self.csv_record_final) - 1):
            component_label = self.csv_record_final[record_idx][self.component_idx]
            component_list[component_label] = ''
        for info in self.barcode_info:
            component_label = info[3]
            component_list[component_label] = ''
        component_list[0] = ''

        max_component_label = max(component_list.keys()) if component_list else 0
        unique_component_count = max(len(component_list.keys()), max_component_label + 1)

        set1 = altairThemes.get_colour_scheme("Set1", unique_component_count)
        colors = set1
        colors[0] = '#000000'

        ids = []
        x_index = []
        y_index = []
        colors_point = []

        for i in range(len(self.barcode_info)):
            ids.append(i)
            x_index.append(self.barcode_info[i][1])
            y_index.append(self.barcode_info[i][2])
            component_label = self.barcode_info[i][3]
            if component_label >= len(colors):
                colors_point.append('#000000')
            else:
                colors_point.append(colors[component_label])

        G = nx.MultiDiGraph(directed=True)

        for i in range(len(self.barcode_info)):
            marker_size = 'circle'
            label_str = f"{i}_c:{self.barcode_info[i][3]}"
            if self.barcode_type.get(self.barcode_info[i][0], ''):
                label_str += f"_{self.barcode_type[self.barcode_info[i][0]]}"
            G.add_node(
                int(ids[i]),
                x=int(x_index[i]),
                y=int(y_index[i]),
                label=label_str,
                pos=f"{x_index[i]},{-y_index[i]} !",
                physics=False,
                shape=marker_size,
                color=matplotlib.colors.rgb2hex(colors_point[i]),
                size=100000
            )

        score_list = [self.csv_record_final[k][self.score_idx] for k in range(1, len(self.csv_record_final) - 1)]
        if score_list:
            min_score = np.min(score_list)
            max_score = np.max(score_list)
        else:
            min_score = 0
            max_score = 1

        count_edges = 0
        for k in range(1, len(self.csv_record_final) - 1):
            i = self.csv_record_final[k][self.from_id_idx]
            j = self.csv_record_final[k][self.to_id_idx]
            ligand = self.csv_record_final[k][self.csv_record_final[0].index('ligand')]
            receptor = self.csv_record_final[k][self.csv_record_final[0].index('receptor')]
            edge_score = self.csv_record_final[k][self.score_idx]
            if max_score > min_score:
                edge_score = (edge_score - min_score) / (max_score - min_score)
            else:
                edge_score = 0

            fi_cluster = self.barcode_type.get(self.barcode_info[i][0], '')
            tj_cluster = self.barcode_type.get(self.barcode_info[j][0], '')
            title_str = f"{fi_cluster}→{tj_cluster} | L:{ligand}, R:{receptor}, {edge_score:.2f}"

            G.add_edge(
                int(i),
                int(j),
                label=title_str,
                color=colors_point[i],
                value=np.float64(edge_score)
            )
            count_edges += 1

        print(f"总边数: {count_edges}")

        nt = Network(directed=True, height='1000px', width='100%')
        nt.from_nx(G)
        network_file = f"{self.output_name}_mygraph.html"
        nt.save_graph(network_file)
        print(f"✅ 网络图形已保存至: {network_file}")

        dot_file = f"{self.output_name}_test_interactive.dot"
        write_dot(G, dot_file)
        print(f"✅ dot文件已保存至: {dot_file}")

    def visualize_intra_domain_network(self, domain_value, min_score=None, outfile_suffix=None):
        """可视化单个域内部的通讯网络（domain/layer/cluster）"""
        if not hasattr(self, "barcode_domain_map") or self.barcode_domain_map is None:
            raise RuntimeError("缺少域映射，请先确保 load_annotations() 已加载含 cluster/layer 的注释表")

        domain_value = str(domain_value)

        chosen_edges = []
        nodes_involved = set()
        for k in range(1, len(self.csv_record_final) - 1):
            rec = self.csv_record_final[k]
            i, j = rec[self.from_id_idx], rec[self.to_id_idx]
            score = float(rec[self.score_idx])
            if min_score is not None and score < float(min_score):
                continue
            bi = str(self.barcode_info[i][0])
            bj = str(self.barcode_info[j][0])
            di = str(self.barcode_domain_map.get(bi, ""))
            dj = str(self.barcode_domain_map.get(bj, ""))
            if di == domain_value and dj == domain_value:
                chosen_edges.append(k)
                nodes_involved.update([i, j])

        print(f"[单域] {domain_value}: 选中 {len(chosen_edges)} 条边，涉及 {len(nodes_involved)} 个节点")

        if not chosen_edges:
            print("⚠️ 没有符合条件的边")
            return

        G = nx.MultiDiGraph(directed=True)
        for i in nodes_involved:
            b = self.barcode_info[i][0]
            x, y = self.barcode_info[i][1], self.barcode_info[i][2]
            G.add_node(
                int(i),
                x=int(x), y=int(y),
                label=f"{i}_{domain_value}",
                pos=f"{x},{-y} !",
                physics=False,
                shape="circle"
            )
        for k in chosen_edges:
            rec = self.csv_record_final[k]
            i, j = rec[self.from_id_idx], rec[self.to_id_idx]
            ligand, receptor = rec[self.csv_record_final[0].index('ligand')], rec[self.csv_record_final[0].index('receptor')]
            score = float(rec[self.score_idx])
            G.add_edge(int(i), int(j), ligand=ligand, receptor=receptor, score=score)

        und = G.to_undirected(as_view=True)
        comps = list(nx.connected_components(und))
        comp_id = {}
        for cid, nodes in enumerate(comps, start=1):
            for n in nodes:
                comp_id[n] = cid

        ncol = max(2, len(comps) + 1)
        palette = altairThemes.get_colour_scheme("Set1", ncol)
        palette[0] = "#000000"
        comp_color = {cid: matplotlib.colors.rgb2hex(palette[cid % len(palette)]) for cid in range(1, len(comps) + 1)}

        for n in G.nodes():
            c = comp_color[comp_id[n]]
            G.nodes[n]["color"] = c
            G.nodes[n]["size"] = 100000

        for u, v, key, data in G.edges(keys=True, data=True):
            data["label"] = f"L:{data['ligand']}, R:{data['receptor']}, {data['score']:.2f}"
            data["color"] = G.nodes[u]["color"]
            data["value"] = data["score"]

        suffix = outfile_suffix or f"intra_domain_{domain_value}"
        nt = Network(directed=True, height="900px", width="100%")
        nt.from_nx(G)
        out_html = f"{self.output_name}_{suffix}.html"
        nt.save_graph(out_html)
        write_dot(G, f"{self.output_name}_{suffix}.dot")
        print(f"✅ 单域网络已保存: {out_html}")

    def visualize_cross_domain_network(self, domain_A, domain_B, bidirectional=True, min_score=None,
                                       outfile_suffix=None):
        """可视化两个域之间的通讯网络（domain/layer/cluster）"""
        if not hasattr(self, "barcode_domain_map") or self.barcode_domain_map is None:
            raise RuntimeError("缺少域映射，请先确保 load_annotations() 已加载含 cluster/layer 的注释表")

        A = str(domain_A)
        B = str(domain_B)

        chosen_edges = []
        nodes_involved = set()

        for k in range(1, len(self.csv_record_final) - 1):
            rec = self.csv_record_final[k]
            i, j = rec[self.from_id_idx], rec[self.to_id_idx]
            score = float(rec[self.score_idx])
            if min_score is not None and score < float(min_score):
                continue
            bi = str(self.barcode_info[i][0])
            bj = str(self.barcode_info[j][0])
            di = str(self.barcode_domain_map.get(bi, ""))
            dj = str(self.barcode_domain_map.get(bj, ""))

            cond_AB = (di == A and dj == B)
            cond_BA = (di == B and dj == A)

            if cond_AB or (bidirectional and cond_BA):
                chosen_edges.append(k)
                nodes_involved.update([i, j])

        print(f"[跨域] {A} ↔ {B}：选中 {len(chosen_edges)} 条边，涉及 {len(nodes_involved)} 个节点")

        if not chosen_edges:
            print("⚠️ 这两个域之间没有符合条件的边")
            return

        G = nx.MultiDiGraph(directed=True)
        for i in nodes_involved:
            b = self.barcode_info[i][0]
            x, y = self.barcode_info[i][1], self.barcode_info[i][2]
            d = str(self.barcode_domain_map.get(str(b), ""))
            G.add_node(
                int(i),
                x=int(x), y=int(y),
                label=f"{i}_{d}",
                pos=f"{x},{-y} !",
                physics=False,
                shape="circle"
            )
        for k in chosen_edges:
            rec = self.csv_record_final[k]
            i, j = rec[self.from_id_idx], rec[self.to_id_idx]
            ligand, receptor = rec[self.csv_record_final[0].index('ligand')], rec[self.csv_record_final[0].index('receptor')]
            score = float(rec[self.score_idx])
            G.add_edge(int(i), int(j), ligand=ligand, receptor=receptor, score=score)

        und = G.to_undirected(as_view=True)
        comps = list(nx.connected_components(und))
        comp_id = {}
        for cid, nodes in enumerate(comps, start=1):
            for n in nodes:
                comp_id[n] = cid

        ncol = max(2, len(comps) + 1)
        palette = altairThemes.get_colour_scheme("Set1", ncol)
        palette[0] = "#000000"
        comp_color = {cid: matplotlib.colors.rgb2hex(palette[cid % len(palette)]) for cid in range(1, len(comps) + 1)}

        for n in G.nodes():
            c = comp_color[comp_id[n]]
            G.nodes[n]["color"] = c
            G.nodes[n]["size"] = 100000

        for u, v, key, data in G.edges(keys=True, data=True):
            data["label"] = f"L:{data['ligand']}, R:{data['receptor']}, {data['score']:.2f}"
            data["color"] = G.nodes[u]["color"]
            data["value"] = data["score"]

        dir_tag = "bi" if bidirectional else "AtoB"
        suffix = outfile_suffix or f"cross_{A}_{B}_{dir_tag}"
        nt = Network(directed=True, height="900px", width="100%")
        nt.from_nx(G)
        out_html = f"{self.output_name}_{suffix}.html"
        nt.save_graph(out_html)
        write_dot(G, f"{self.output_name}_{suffix}.dot")
        print(f"✅ 跨域网络已保存: {out_html}")

    def run(self):
        """执行完整的可视化流程"""
        print("\n" + "=" * 70)
        print("开始 PlantCCC 可视化（单尺度）")
        print("=" * 70 + "\n")

        self.load_barcode_info()
        self.load_selfloop_info()
        self.load_annotations()
        self.load_ccc_data()

        self.preprocess_ccc_data()
        self.find_connected_components()
        self.filter_ccc_data()
        self.save_ccc_list()

        self.prepare_active_spots()

        self.generate_component_plot()
        self.generate_histograms()
        self.generate_attention_histogram()
        self.generate_network_graph()

        print("\n" + "=" * 70)
        print("✅ 所有可视化任务完成！")
        print("=" * 70)


def main():
    """主函数"""
    visualizer = PlantCCCVisualizer(
        data_name="Arabidopsis",
        model_name="Arabidopsis_model",
        top_edge_count=3000,
        top_ccc_file="output/IN11heng_200+L_R/IN11heng_model_top20percent.csv",
        metadata_from="metadata/",
        output_path="output/Arabidopsis",
        filter=-1,
        filter_by_ligand_receptor=['Pop_G13G072704-Pop_G03G078104'],
        filter_by_annotation=[],
        sort_by_attentionScore=1
    )
    visualizer.run()


if __name__ == "__main__":
    main()
