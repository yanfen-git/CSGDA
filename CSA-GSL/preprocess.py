# preprocess_smote.py
import os
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
import dgl
from sklearn.preprocessing import StandardScaler


def normalize_adj(adj: sp.coo_matrix) -> sp.coo_matrix:
    """
    对邻接矩阵做对称归一化:  D^{-1/2} A D^{-1/2}
    adj: SciPy 稀疏矩阵 (coo / csr / csc)
    """
    adj = adj.tocoo()
    # 加自环
    adj = adj + sp.eye(adj.shape[0], dtype=adj.dtype, format="coo")
    rowsum = np.array(adj.sum(1))  # (N, 1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def _read_cellstatus_csv(split: str, drug: str) -> pd.DataFrame:
    assert split in ["source", "target"]
    drug_lower = drug.lower()

    # ===== PLX4720 (2-1 / 2-2) =====
    if drug_lower == "plx4720":
        base_dir = "data/share_data/1"
        if split == "source":
            fname = "Sc_matrix.PLX4720_GSE108383_451Lu_withPseudoLabel.csv"
        else:
            fname = "Sc_matrix.PLX4720_GSE108394_451Lu_withPseudoLabel.csv"
        path = os.path.join(base_dir, fname)

    # ===== Paclitaxel + Atezolizumab (2-3) =====
    elif drug_lower == "paclitaxel+atezolizumab":
        base_dir = "data/share_data/3"
        if split == "source":
            fname = "Sc_matrix.paclitaxel+atezolizumab_GSE169246_PacAteTissue_P012pre+P002pre.csv"
        else:
            fname = "Sc_matrix.paclitaxel+atezolizumab_GSE169246_PacAteBlood_P012pre+P002pre.csv"
        path = os.path.join(base_dir, fname)

    # ===== Paclitaxel (2-4) =====
    elif drug_lower == "paclitaxel":
        base_dir = "data/share_data/4"
        if split == "source":
            fname = "Sc_matrix.paclitaxel_GSE169246_PacTissue_P022pre+P025pre_withPseudoLabel.csv"
        else:
            fname = "Sc_matrix.paclitaxel_GSE169246_PacBlood_P022pre+P025pre_withPseudoLabel.csv"
        path = os.path.join(base_dir, fname)

    # ===== Cisplatin (2-5, Primary / Metastatic) =====
    elif drug_lower == "cisplatin":
        base_dir = "data/share_data/5"
        if split == "source":   # Primary 当作 source
            fname = "GSE117872_Cisplatin_Primary_withLabel_withPseudoLabel.csv"
        else:                   # Metastatic 当作 target
            fname = "GSE117872_Cisplatin_Metastatic_withLabel_withPseudoLabel.csv"
        path = os.path.join(base_dir, fname)

    elif drug_lower == "erlotinib":
        base_dir = "data/share_data/6"
        if split == "source":
            fname = "GSE149383_erlotinib_bulk_withLabel.csv"
        else:  # target
            fname = "GSE112274_erlotinib_sc_withPseudoLabel.csv"
        path = os.path.join(base_dir, fname)

    else:
        raise ValueError(f"暂时只在 _read_cellstatus_csv 里实现了这些 drug，当前 drug={drug}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    print(f"[INFO] 读取 {split} 数据: {path}")
    df = pd.read_csv(path, index_col=0)
    return df


def load_data_cellStatus(split: str, drug: str, k: int = 15):
    """
    从 csv 里面读出特征 + 标签 + kNN 图（用于 ProGNN）
    - 优先用 pseudo_label，当标签；
    - 如果没有 pseudo_label，就退回用 response。
    """
    assert split in ["source", "target"]

    # 1) 读 csv（你之前已经实现好的 _read_cellstatus_csv）
    df = _read_cellstatus_csv(split, drug)

    # ===== 选标签列：优先 pseudo_label，没有就用 response =====
    if "pseudo_label" in df.columns:
        label_col = "pseudo_label"
    elif "response" in df.columns:
        label_col = "response"
    else:
        raise ValueError(
            f"{split} 数据中既没有 'pseudo_label' 也没有 'response' 列，请检查文件。"
        )

    # ---------- 1. 特征矩阵 X：去掉标签列 ----------
    drop_cols = [c for c in ["response", "pseudo_label"] if c in df.columns]
    feature_df = df.drop(columns=drop_cols)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_df.values)
    features = torch.FloatTensor(features_scaled)

    # ---------- 2. 标签 y ----------
    raw_labels = df[label_col]

    # 如果本来就是数字，就直接转 int64
    if pd.api.types.is_numeric_dtype(raw_labels):
        labels_np = raw_labels.astype("int64").to_numpy()
    else:
        # 比如 'Hypoxia', 'EMT' 之类字符串，就做一个类别编码
        codes, uniques = pd.factorize(raw_labels)
        mapping = {str(u): int(i) for i, u in enumerate(uniques)}
        print("[INFO] 标签为非数值类型，已自动映射为整数标签：")
        print("       " + ", ".join([f"{k} -> {v}" for k, v in mapping.items()]))
        labels_np = codes.astype("int64")

    labels = torch.LongTensor(labels_np)

    # ---------- 3. 用 DGL 构 kNN 图 ----------
    knn_graph = dgl.knn_graph(features, k=k, algorithm="kd-tree", dist="cosine")
    knn_graph = dgl.add_self_loop(knn_graph)

    # DGL 稀疏邻接 -> SciPy COO
    adj_torch = knn_graph.adjacency_matrix(transpose=False).coalesce()
    idx = adj_torch.indices().cpu().numpy()
    vals = adj_torch.values().cpu().numpy()
    row, col = idx
    sp_adj = sp.coo_matrix(
        (vals, (row, col)),
        shape=(knn_graph.num_nodes(), knn_graph.num_nodes()),
        dtype=np.float32,
    )

    # 归一化邻接矩阵
    adj_normalized = normalize_adj(sp_adj)
    n_features = features.shape[1]

    # 一点 sanity check
    if torch.isnan(features).any():
        print("[WARN] features 中存在 NaN")
    if torch.isnan(labels.float()).any():
        print("[WARN] labels 中存在 NaN")

    return adj_normalized, features, labels, knn_graph, n_features