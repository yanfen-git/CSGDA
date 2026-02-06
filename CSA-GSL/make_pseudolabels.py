import os
import pandas as pd
import numpy as np

# ------- 1. 读取状态基因集（用 GeneName 那一列） -------

def load_state_gene_dict(status_dir="./data/cellStatusCancerSEA"):
    """
    每个 txt 文件格式：
    EnsembleID   GeneName
    ENSG...      APAF1
    ...

    我们只取 GeneName 列，统一转成大写。
    """
    state_gene_dict = {}

    for fname in os.listdir(status_dir):
        if not fname.endswith(".txt"):
            continue

        status = fname.replace(".txt", "")
        path = os.path.join(status_dir, fname)

        # 用制表符或空白分隔读取
        df = pd.read_csv(path, sep=r"\s+", engine="python")

        # 找到 GeneName 那一列
        if "GeneName" in df.columns:
            gene_col = "GeneName"
        else:
            # 如果列名不叫 GeneName，就取最后一列做基因名
            gene_col = df.columns[-1]

        genes = (
            df[gene_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .dropna()
            .unique()
            .tolist()
        )

        state_gene_dict[status] = genes
        print(f"[STATE] {status}: 从 {fname} 读到 {len(genes)} 个基因名")

    print("\n状态列表:", list(state_gene_dict.keys()))
    return state_gene_dict


# ------- 2. 对一个表达矩阵计算状态得分 + 伪标签 -------

def compute_scores_and_labels_for_file(csv_path, state_gene_dict):
    """
    csv_path: Sc_matrix....csv
    假设:
        - index = 细胞ID
        - 第一列 = response
        - 其余列 = 基因表达（列名是基因符号）
    """
    print(f"\n[INFO] 处理文件: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)

    # 拆出 response 和 基因表达矩阵
    response = df.iloc[:, 0]
    expr = df.iloc[:, 1:]
    print(f"[INFO] 细胞数: {expr.shape[0]}, 基因数: {expr.shape[1]}")

    # 矩阵基因列名（同时准备大写版方便匹配）
    gene_cols = list(expr.columns)
    gene_cols_upper = [g.upper() for g in gene_cols]

    scores = pd.DataFrame(index=expr.index)

    for status, genes_upper in state_gene_dict.items():
        # 在矩阵列名中寻找这些基因
        keep_idx = [i for i, g in enumerate(gene_cols_upper) if g in genes_upper]

        if len(keep_idx) == 0:
            print(f"[WARN] 状态 {status} 在该矩阵中没有匹配到基因，得分全 0")
            scores[status] = 0.0
            continue

        matched_genes = [gene_cols[i] for i in keep_idx]
        print(f"[INFO] 状态 {status} 匹配到 {len(matched_genes)} 个基因")

        sub_expr = expr[matched_genes]          # 细胞 × 匹配到的基因
        scores[status] = sub_expr.mean(axis=1)  # 对基因求行平均 → 每个细胞一个分数

    # 伪标签 = 得分最大的状态
    pseudo_label = scores.idxmax(axis=1)
    scores["pseudo_label"] = pseudo_label

    return df, scores


# ------- 3. 主程序：只处理 share_data/1 和 share_data/2 -------

if __name__ == "__main__":
    # 1. 读取基因集
    state_gene_dict = load_state_gene_dict("./data/cellStatusCancerSEA")

    # 现在处理 1、2、3、4 四个文件夹
    subdirs = [
        "./data/share_data/5",
    ]

    for subdir in subdirs:
        if not os.path.isdir(subdir):
            continue

        for fname in os.listdir(subdir):
            if not fname.endswith(".csv"):
                continue
            # 不再按药物筛选，所有 Sc_matrix.xxx 都处理
            in_path = os.path.join(subdir, fname)

            # 计算状态得分 + 伪标签
            raw_df, score_df = compute_scores_and_labels_for_file(in_path, state_gene_dict)

            base = fname.replace(".csv", "")

            # 1) 保存状态得分
            scores_out = os.path.join(subdir, base + "_stateScores.csv")
            score_df.to_csv(scores_out)
            print(f"[OK] 状态得分保存到: {scores_out}")

            # 2) 保存带伪标签的表达矩阵
            out_df = raw_df.copy()
            out_df["pseudo_label"] = score_df["pseudo_label"]
            matrix_out = os.path.join(subdir, base + "_withPseudoLabel.csv")
            out_df.to_csv(matrix_out)
            print(f"[OK] 带伪标签的表达矩阵保存到: {matrix_out}")

    print("\n[Done] share_data/5 中所有 Sc_matrix.*.csv 都已生成伪标签。")
