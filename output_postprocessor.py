import numpy as np
import pickle
import statistics
from scipy.stats import skew
from collections import defaultdict
import pandas as pd
import gzip
import gc
import os


class PlantCCCPostProcessor:
    """
    - 生成细胞通信分析结果
    """

    def __init__(self, data_name, model_name,
                 embedding_path='embedding_data/',
                 metadata_from='metadata/',
                 data_from='input_graph/',
                 output_path='output/',
                 top_percent=20):

        self.data_name = data_name
        self.model_name = model_name
        self.embedding_path = os.path.join(embedding_path, data_name)
        self.metadata_from = os.path.join(metadata_from, data_name)
        self.data_from = os.path.join(data_from, data_name)
        self.output_path = os.path.join(output_path, data_name)
        self.top_percent = top_percent

        self.barcode_info = None
        self.lig_rec_dict = None
        self.row_col = None  # shape: (E,2) 按预处理保存顺序
        self.lig_rec = None  # len=E, 每条边的 (ligand,receptor)
        self.total_num_cell = None

        self.results = {
            'scores': [],
            'edges': [],
            'raw_scores': {},
            'lr_pairs': set()
        }

        self._create_output_dir()
        self._validate_attention_file()

    def _create_output_dir(self):
        """创建输出目录"""
        os.makedirs(self.output_path, exist_ok=True)

    def _attention_file(self):
        """获取注意力文件路径"""
        return f"{self.embedding_path}/{self.model_name}_attention"

    def _validate_attention_file(self):
        """验证注意力文件是否存在"""
        print("\n" + "=" * 70)
        print("验证GAT输出文件...")
        print("=" * 70)

        attention_file = self._attention_file()
        if os.path.exists(attention_file):
            print(f"✅ 找到注意力文件: {os.path.basename(attention_file)}")
        else:
            raise FileNotFoundError(
                f"未找到 注意力文件: {attention_file}\n"
                f"请确认训练阶段输出了 {self.model_name}_attention 文件"
            )

        print("=" * 70 + "\n")

    def load_metadata(self):
        """加载元数据与L-R邻接信息（同时保留 edge-level 有序记录）"""
        print("加载元数据...")

        barcode_file = os.path.join(self.metadata_from, f"{self.data_name}_barcode_info")
        with gzip.open(barcode_file, 'rb') as fp:
            self.barcode_info = pickle.load(fp)
        print(f"✅ 加载了 {len(self.barcode_info)} 个细胞")

        adjacency_file = os.path.join(self.data_from, f"{self.data_name}_adjacency_records")
        if not os.path.exists(adjacency_file):
            raise FileNotFoundError(f"未找到邻接记录文件: {adjacency_file}")

        print("加载邻接记录...")
        with gzip.open(adjacency_file, 'rb') as fp:
            row_col, edge_weight, lig_rec, total_num_cell = pickle.load(fp)

        # ✅ 关键：保留“有序”的 edge-level 记录（用于一一对齐 attention）
        self.row_col = np.asarray(row_col, dtype=np.int64)  # (E,2)
        self.lig_rec = list(lig_rec)  # len=E, [(lig,rec),...]
        self.total_num_cell = int(total_num_cell)

        # （可选）仍然保留你原来的 lig_rec_dict：仅用于 component/self-loop 判断
        self.lig_rec_dict = defaultdict(lambda: defaultdict(list))
        for idx in range(len(row_col)):
            i = row_col[idx][0]
            j = row_col[idx][1]
            lr_pair = lig_rec[idx]
            if lr_pair not in self.lig_rec_dict[i][j]:
                self.lig_rec_dict[i][j].append(lr_pair)

        print(f"✅ 加载了 {len(self.row_col)} 条边（edge-level）")
        print(f"✅ 涉及 {len(self.lig_rec_dict)} 个发送细胞\n")

        del row_col, edge_weight, lig_rec
        gc.collect()

    def process_attention(self):
        """严格修复版（P0-2）：按论文语义对齐
        - edge-level 严格对齐：attention 与 (i,j,ligand,receptor) 按 edge idx 一一对应
        - CCC 分数使用 Eq.(1) 的 unnormalized attention（全局 min-max 到 [0,1]）
        - softmax attention（Eq.(2)）仅用于 debug，不用于 CCC ranking
        """
        from collections import defaultdict
        import gzip, pickle
        import numpy as np

        print("=" * 70)
        print("处理 GAT 注意力权重（edge-level 严格对齐版，P0-2语义对齐）...")
        print("=" * 70)

        if self.row_col is None or self.lig_rec is None:
            raise RuntimeError("请先运行 load_metadata()，确保已加载 row_col/lig_rec。")

        attention_file = self._attention_file()

        with gzip.open(attention_file, 'rb') as fp:
            attention_bundle = pickle.load(fp)

        # -------- 小工具：兼容 torch / numpy / list --------
        def _to_numpy(x):
            try:
                import torch
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
            except Exception:
                pass
            if isinstance(x, np.ndarray):
                return x
            return np.asarray(x)

        def _reduce_to_edge_vector(att, E, name="attention"):
            """把 attention 规整成 shape=(E,) 的 per-edge 标量向量，并处理 multi-head."""
            a = _to_numpy(att)
            a = np.squeeze(a)

            if a.ndim == 0:
                # 单个标量不合理
                raise ValueError(f"{name} 只有一个标量，无法对应 E={E} 条边。shape={getattr(att, 'shape', None)}")

            if a.ndim == 1:
                if a.shape[0] != E:
                    raise ValueError(f"{name} 维度不匹配：len={a.shape[0]} vs E={E}。shape={a.shape}")
                return a.astype(np.float64)

            if a.ndim == 2:
                # 常见两种：[E, H] 或 [H, E] 或 [E,1]
                if a.shape[0] == E:
                    return a.mean(axis=1).astype(np.float64)
                if a.shape[1] == E:
                    return a.mean(axis=0).astype(np.float64)
                raise ValueError(f"{name} 是二维但无法判断哪一维是 E：shape={a.shape}, E={E}")

            # 更高维：尽量 squeeze 后仍 >2 说明结构复杂，直接报错更安全
            raise ValueError(f"{name} 维度过高（>2），请检查训练端保存格式：shape={a.shape}")

        # -------- 解析 bundle --------
        if len(attention_bundle) >= 6:
            idx_l2 = attention_bundle[5]  # edge_index for layer2
            att_eq1 = attention_bundle[2]  # Eq.(1) unnormalized attention（或其logit）
            att_softmax = attention_bundle[4]  # Eq.(2) softmax-normalized alpha（仅debug）
            edge_indices = _to_numpy(idx_l2)
            print("✅ 使用 bundle 新格式：idx_l2 + attention")
        else:
            raise ValueError("attention_bundle 格式过旧/不完整，不支持严格对齐。")

        # edge_indices 规整成 (2, E)
        edge_indices = np.asarray(edge_indices)
        if edge_indices.shape[0] != 2:
            # 有些实现是 (E,2)，这里做一次转置兜底
            if edge_indices.shape[1] == 2:
                edge_indices = edge_indices.T
            else:
                raise ValueError(f"edge_indices 形状异常，期望 (2,E) 或 (E,2)，实际 {edge_indices.shape}")

        E_att = edge_indices.shape[1]
        E_meta = len(self.lig_rec)

        # -------- 关键一致性检查（非常重要）--------
        if E_att != E_meta:
            raise ValueError(
                f"❌ edge 数不一致：attention边数={E_att}, adjacency_records边数={E_meta}。\n"
                f"这通常是训练时 GAT 层内部 add_self_loops/coalesce 改变了边集合或顺序。\n"
                f"解决方案：训练端保存 edge_id 并按 edge_id 对齐，或禁用会改变边集合/顺序的操作。"
            )

        meta_edge_index = np.asarray(self.row_col).T  # (2,E)
        if not np.array_equal(meta_edge_index, edge_indices):
            raise ValueError(
                "❌ edge_index 顺序与 adjacency_records 不一致，无法保证 lig_rec[idx] 的严格对齐。\n"
                "这意味着训练过程中边顺序发生了重排（常见原因：add_self_loops + coalesce）。\n"
                "请在训练端做硬修复：保存 edge_id 并按 edge_id 对齐后再保存 attention。"
            )

        # -------- P0-2 核心：CCC 分数用 Eq.(1) unnormalized attention --------
        att_u = _reduce_to_edge_vector(att_eq1, E_att, name="att_eq1 (Eq.1 unnormalized)")
        att_n = _reduce_to_edge_vector(att_softmax, E_att, name="att_softmax (Eq.2 normalized)")

        # Eq.(1) 理论上经过 tanh 应在 [-1,1]。若明显超出，说明这里更像 logit，补 tanh()
        if (att_u.max() > 1.0001) or (att_u.min() < -1.0001):
            att_u = np.tanh(att_u)
            print("ℹ️ 检测到 Eq.(1) attention 超出[-1,1]，已在后处理中补 tanh() 以对齐论文公式。")
        else:
            print("ℹ️ Eq.(1) attention 已在[-1,1]，默认认为训练端已做 tanh()。")

        # 全局 min-max 到 [0,1]：论文的 communication probability / ranking 语义
        smin, smax = float(att_u.min()), float(att_u.max())
        den = (smax - smin) if (smax > smin) else 1.0
        scaled_all = (att_u - smin) / den

        print(f"✅ CCC 使用 Eq.(1) unnormalized attention（补tanh后）并全局缩放到[0,1]")
        print(
            f"   Eq.(1)范围: [{att_u.min():.4f}, {att_u.max():.4f}] -> scaled范围: [{scaled_all.min():.4f}, {scaled_all.max():.4f}]")
        print(f"   Eq.(2) softmax范围(仅debug): [{att_n.min():.4f}, {att_n.max():.4f}]")

        # -------- 严格一一对应：第 idx 条边 -> lig_rec[idx] --------
        communication_dict = defaultdict(list)
        lr_pairs_set = set()

        for idx in range(E_att):
            i = int(edge_indices[0, idx])
            j = int(edge_indices[1, idx])

            ligand, receptor = self.lig_rec[idx]  # ✅ 关键：同 idx 的 lig_rec
            scaled = float(scaled_all[idx])

            key = f"{i}+{j}+{ligand}+{receptor}"
            communication_dict[key].append(scaled)
            lr_pairs_set.add((ligand, receptor))

        print(f"✅ 有效通信键数: {len(communication_dict)}（理论上≈边数E={E_att}）")

        # -------- 生成 edge_list（rank 按 score 降序）--------
        tmp = []
        raw_scores_dict = {}

        for key, scores in communication_dict.items():
            sc = float(np.mean(scores))
            tmp.append((key, sc))
            raw_scores_dict[key] = sc

        tmp.sort(key=lambda x: x[1], reverse=True)

        edge_list = []
        for rk, (k, sc) in enumerate(tmp, start=1):
            edge_list.append([k, rk, sc])

        self.results['edges'] = edge_list
        self.results['scores'] = [e[2] for e in edge_list]
        self.results['raw_scores'] = raw_scores_dict
        self.results['lr_pairs'] = lr_pairs_set

        # 可选：把 debug 信息也存一下，方便你排查
        self.results['debug_att_eq1_scaled'] = scaled_all.tolist()
        self.results['debug_att_eq1_raw'] = att_u.tolist()
        self.results['debug_att_softmax_mean'] = att_n.tolist()

        print("✅ 处理完成：")
        print(f"   通信数: {len(edge_list)}")
        print(f"   L–R对种类: {len(lr_pairs_set)}")
        print(f"   分数范围: [{min(self.results['scores']):.4f}, {max(self.results['scores']):.4f}]")

    def save_results(self):
        """保存结果（恢复旧版CSV表头与component计算）
        输出列：
          from_cell,to_cell,ligand,receptor,edge_rank,component,from_id,to_id,attention_score
        """
        import os
        import numpy as np
        import pandas as pd
        from collections import defaultdict
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        print("\n" + "=" * 70)
        print("保存结果（旧版表头）.")
        print("=" * 70)

        edge_list = self.results['edges']  # [key, rank, score] 其中 score∈[0,1]，已等价旧 attention_score
        lr_pairs = self.results.get('lr_pairs', set())

        if not edge_list:
            print("⚠️ 无有效数据可保存")
            return

        # ---------------- 计算 component（按旧版逻辑） ----------------
        n = len(self.barcode_info)
        connecting = np.zeros((n, n), dtype=int)

        # 任一 (i,j, L–R) 存在即视为有边
        for key, _, _ in edge_list:
            i, j, _, _ = key.split('+')
            i = int(i);
            j = int(j)
            connecting[i, j] = 1

        # weak 连接分量
        graph = csr_matrix(connecting)
        n_comp, labels = connected_components(csgraph=graph, directed=True, connection='weak', return_labels=True)

        # 各分量内点数
        counts = np.zeros(n_comp, dtype=int)
        for lab in labels:
            counts[lab] += 1

        # 多点分量编号从2开始
        comp_map = {}
        next_id = 2
        for cid in range(n_comp):
            if counts[cid] > 1:
                comp_map[cid] = next_id
                next_id += 1

        # 给每个 cell 赋 component id：多点→2,3,...；自分泌→1；其他→0
        cell_component = [0] * n
        for i in range(n):
            if counts[labels[i]] > 1:
                cell_component[i] = comp_map[labels[i]]
            elif connecting[i, i] == 1 and (
                    i in self.lig_rec_dict and i in self.lig_rec_dict[i] and len(self.lig_rec_dict[i][i]) > 0):
                cell_component[i] = 1
            else:
                cell_component[i] = 0

        # ---------------- 组装CSV（恢复旧版列名与含义） ----------------
        header = ['from_cell', 'to_cell', 'ligand', 'receptor', 'edge_rank', 'component', 'from_id', 'to_id',
                  'attention_score']
        records_all = [header]

        # 也顺便做个 L-R 统计（不影响主CSV）
        lr_stats = defaultdict(lambda: {'count': 0, 'total_score': 0.0})

        for key, rank, score in edge_list:
            i_str, j_str, ligand, receptor = key.split('+')
            i = int(i_str);
            j = int(j_str)

            comp_val = cell_component[i]
            if comp_val == 0:
                # 旧版遇到0会打印error并跳过写出；保持兼容
                # print('warning: component=0 at from_id', i)
                continue
            comp_field = '0-single' if comp_val == 1 else comp_val

            records_all.append([
                self.barcode_info[i][0],  # from_cell
                self.barcode_info[j][0],  # to_cell
                ligand,
                receptor,
                int(rank),  # edge_rank（1是最高）
                comp_field,  # component
                i,  # from_id
                j,  # to_id
                float(score)  # attention_score（0~1）
            ])

            lr_stats[(ligand, receptor)]['count'] += 1
            lr_stats[(ligand, receptor)]['total_score'] += float(score)

        # 保存 allCCC（旧名与新名并存你可按需保留）
        out_all = os.path.join(self.output_path, f"{self.model_name}_allCCC.csv")
        pd.DataFrame(records_all[1:], columns=records_all[0]).to_csv(out_all, index=False)
        print(f"\n✅ 已保存: {os.path.basename(out_all)} ({len(records_all) - 1} 条)")

        # 保存 Top N%
        top_n = max(1, int(len(records_all[1:]) * self.top_percent / 100))
        out_top = os.path.join(self.output_path, f"{self.model_name}_top{self.top_percent}percent.csv")
        pd.DataFrame(records_all[1:top_n + 1], columns=records_all[0]).to_csv(out_top, index=False)
        print(f"✅ Top {self.top_percent}%: {os.path.basename(out_top)} ({top_n} 条)")

        # （可选）保留你当前版本里对 L-R 的统计与去重列表
        lr_pairs_file = os.path.join(self.output_path, f"{self.model_name}_unique_LR_pairs.csv")
        with open(lr_pairs_file, 'w') as f:
            f.write("ligand,receptor\n")
            for ligand, receptor in sorted(lr_pairs):
                f.write(f"{ligand},{receptor}\n")
        print(f"✅ L-R对列表: {os.path.basename(lr_pairs_file)} ({len(lr_pairs)} 种)")

        # 详细统计（数量、均分等）
        lr_stats_rows = []
        for (lig, rec), info in lr_stats.items():
            cnt = info['count']
            tot = info['total_score']
            lr_stats_rows.append({
                'Ligand': lig,
                'Receptor': rec,
                'Communication_Count': cnt,
                'Avg_Score': f"{(tot / cnt) if cnt > 0 else 0:.6f}",
                'Total_Score': f"{tot:.6f}"
            })
        lr_stats_df = pd.DataFrame(lr_stats_rows).sort_values('Communication_Count', ascending=False)
        lr_stats_file = os.path.join(self.output_path, f"{self.model_name}_LR_pairs_statistics.csv")
        lr_stats_df.to_csv(lr_stats_file, index=False)
        print(f"✅ L-R对详细统计: {os.path.basename(lr_stats_file)} ({len(lr_stats_rows)} 种)")

    def generate_statistics(self):
        """生成统计报告"""
        print("\n" + "=" * 70)
        print("生成统计报告...")
        print("=" * 70)

        scores = self.results['scores']
        lr_pairs = self.results['lr_pairs']

        if len(scores) == 0:
            print("⚠️ 无数据可统计")
            return

        stats_path = os.path.join(self.output_path, f"{self.model_name}_statistics.txt")
        with open(stats_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("PlantCCC 统计报告\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"数据集: {self.data_name}\n")
            f.write(f"模型名: {self.model_name}\n")
            f.write(f"细胞数: {len(self.barcode_info)}\n\n")

            f.write("统计:\n")
            f.write(f"  通信数: {len(scores)}\n")
            f.write(f"  涉及的L-R对数: {len(lr_pairs)}\n")
            f.write(f"  分数范围: [{np.min(scores):.6f}, {np.max(scores):.6f}]\n")
            f.write(f"  中位数: {statistics.median(scores):.6f}\n")
            f.write(f"  平均值: {np.mean(scores):.6f}\n")
            f.write(f"  标准差: {np.std(scores):.6f}\n")
            f.write(f"  偏度: {skew(scores):.6f}\n\n")

            f.write("=" * 70 + "\n")

        print(f"✅ 统计报告已保存: {os.path.basename(stats_path)}")

        # 控制台输出
        print(f"\n📊  统计摘要:")
        print(f"   通信数: {len(scores)}")
        print(f"   L-R对数: {len(lr_pairs)}")
        print(f"   分数范围: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
        print(f"   平均分数: {np.mean(scores):.4f}")

    def run(self):
        """完整流程"""
        print("\n" + "=" * 70)
        print("开始 PlantCCC 后处理")
        print("=" * 70 + "\n")

        self.load_metadata()
        self.process_attention()
        self.save_results()
        self.generate_statistics()

        print("\n" + "=" * 70)
        print("✅ 后处理完成！")
        print("=" * 70)


def main():
    RAW_DATA_ROOT = "../data/Arabidopsis/binned_outputs"
    DATA_NAME = 'square_016um'  # 不要带斜杠
    MODEL_NAME = "Arabidopsis_model"
    # 输出根目录
    OUTPUT_ROOT = "output/Arabidopsis_results"

    # 自动构建子目录 (确保各模块读写路径一致)
    PATH_CONF = {
        'metadata': os.path.join(OUTPUT_ROOT, 'metadata'),
        'input_graph': os.path.join(OUTPUT_ROOT, 'input_graph'),
        'embedding': os.path.join(OUTPUT_ROOT, 'embedding_data'),
        'model': os.path.join(OUTPUT_ROOT, 'model'),
        'vis_output': os.path.join(OUTPUT_ROOT, 'visualization')
    }

    # 定义保留前百分之多少的边
    TOP_PERCENT = 20

    post_processor = PlantCCCPostProcessor(
        data_name=DATA_NAME,
        model_name=MODEL_NAME,
        embedding_path=PATH_CONF['embedding'],  # 读取 Trainer 的输出
        metadata_from=PATH_CONF['metadata'],  # 读取 Preprocessor 的输出
        data_from=PATH_CONF['input_graph'],  # 读取 Preprocessor 的输出
        output_path=PATH_CONF['vis_output'],  # 结果保存位置
        top_percent=TOP_PERCENT
    )

    post_processor.run()


if __name__ == "__main__":

    main()
