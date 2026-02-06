#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
利用 GSE117872 数据构建 Cisplatin 的
- Source / Reference：Primary 细胞
- Target / Query：   Metastatic 细胞
并用 cluster (Sensitive / Resistant) 作为药物反应标签。

输入：
  - GSE117872_good_Data_cellinfo.txt
  - GSE117872_good_Data_TPM.txt

输出：
  - GSE117872_Cisplatin_Primary_withLabel.csv
  - GSE117872_Cisplatin_Metastatic_withLabel.csv
"""

import os
import pandas as pd

# ======== 1. 路径设置 ==========
meta_path = "/root/autodl-tmp/myproject/data/GSE117872_good_Data_cellinfo.txt"
expr_path = "/root/autodl-tmp/myproject/data/GSE117872_good_Data_TPM.txt"
out_dir   = "./"

# ======== 2. 读入 metadata ==========
print(">> 读取 metadata ...")
meta = pd.read_csv(meta_path, sep="\t")

# columns: groups, cell_color, patient_id, origin, drug_status, cluster, ...
print("Metadata columns:", meta.columns.tolist())

# 把细胞ID列统一叫 cell_id，后面会用
meta = meta.rename(columns={"groups": "cell_id"})

# ======== 3. 只保留 Sensitive / Resistant 细胞 ==========
if "cluster" not in meta.columns:
    raise ValueError("metadata 中找不到 'cluster' 列")

valid_resp = ["Sensitive", "Resistant"]
meta = meta[meta["cluster"].isin(valid_resp)].copy()
print("保留 Sensitive/Resistant 后的细胞数:", meta.shape[0])
print(meta["cluster"].value_counts())

# ======== 4. Primary / Metastatic 分组 + 具体 Cisplatin 组合 ==========
# 原发 / 转移：在 drug_status 里
if "drug_status" not in meta.columns:
    raise ValueError("metadata 中找不到 'drug_status' 列")

print("drug_status 分布:\n", meta["drug_status"].value_counts())

# 具体的 Cisplatin 组合：在 cell_color 里
if "cell_color" not in meta.columns:
    raise ValueError("metadata 中找不到 'cell_color' 列")

primary_groups = [
    "HN120P", "HN120PCR",
    "HN137P", "HN137PCR",
    "HN148P",
]

metastatic_groups = [
    "HN120M",  "HN120MCR",
    "HN137M",  "HN137MCR",
    "HN148M",
]

meta_primary = meta[
    (meta["drug_status"] == "Primary") &
    (meta["cell_color"].isin(primary_groups))
].copy()

meta_metastatic = meta[
    (meta["drug_status"] == "Metastatic") &
    (meta["cell_color"].isin(metastatic_groups))
].copy()

print("Primary 细胞数:", meta_primary.shape[0])
print("Metastatic 细胞数:", meta_metastatic.shape[0])

if meta_primary.empty or meta_metastatic.empty:
    print("⚠️ 警告：某一组细胞数为 0，请先检查 cell_color / drug_status 是否和脚本里一致。")

# 看一眼这些组里 origin 分布（只是 sanity check，不做过滤）
if "origin" in meta.columns:
    print("Primary origin 分布:\n", meta_primary["origin"].value_counts())
    print("Metastatic origin 分布:\n", meta_metastatic["origin"].value_counts())

# ======== 5. 读入 TPM 表达矩阵 ==========
print(">> 读取表达矩阵 (TPM) ... 这一步可能稍慢")
expr = pd.read_csv(expr_path, sep="\t", index_col=0)
expr.columns = expr.columns.astype(str)
print("表达矩阵维度: genes x cells =", expr.shape)

# 简单打印前几个细胞名和 cell_id 看看是否对应
print("表达矩阵前10个细胞列名:", expr.columns[:10].tolist())
print("metadata 前10个 cell_id:", meta['cell_id'].astype(str).head(10).tolist())

# ======== 6. 从表达矩阵中截取对应细胞，并打标签 ==========
def build_dataset(meta_subset: pd.DataFrame, expr_all: pd.DataFrame) -> pd.DataFrame:
    """
    根据 metadata 子集，从表达矩阵中截取相应 cell，并在前面加上 cell_id 和 response 列。
    response：Sensitive -> 1, Resistant -> 0
    """
    cell_ids = meta_subset["cell_id"].astype(str).tolist()
    # 只保留表达矩阵里确实存在的细胞
    common_cells = [cid for cid in cell_ids if cid in expr_all.columns]

    print("  >> 该子集 metadata 细胞数:", len(cell_ids),
          "，在表达矩阵中实际找到的细胞数:", len(common_cells))

    if len(common_cells) == 0:
        raise ValueError("在表达矩阵中找不到任何匹配的细胞，请检查 cell_id 是否一致（注意大小写/空格等）。")

    # genes x cells -> cells x genes
    expr_sub = expr_all[common_cells].T
    expr_sub.index.name = "cell_id"

    # 让 metadata 的顺序跟表达矩阵完全一致
    meta_sub = meta_subset.set_index("cell_id").loc[common_cells]

    # 用 cluster 做药物反应标签：Sensitive=1, Resistant=0
    response = (meta_sub["cluster"] == "Sensitive").astype(int)
    expr_sub.insert(0, "response", response.values)

    # 如果想保留 cell_color / drug_status 等信息，也可以加进去（按需启用）
    # expr_sub.insert(1, "cell_color", meta_sub["cell_color"].values)
    # expr_sub.insert(2, "drug_status", meta_sub["drug_status"].values)

    return expr_sub.reset_index()

print(">> 构建 Primary (Source / Reference) 数据集 ...")
df_primary = build_dataset(meta_primary, expr)

print(">> 构建 Metastatic (Target / Query) 数据集 ...")
df_metastatic = build_dataset(meta_metastatic, expr)

print("Primary 数据维度 (cells x features):", df_primary.shape)
print("Metastatic 数据维度 (cells x features):", df_metastatic.shape)

# ======== 7. 保存结果 ==========
os.makedirs(out_dir, exist_ok=True)

out_primary = os.path.join(out_dir, "GSE117872_Cisplatin_Primary_withLabel.csv")
out_meta    = os.path.join(out_dir, "GSE117872_Cisplatin_Metastatic_withLabel.csv")

df_primary.to_csv(out_primary, index=False)
df_metastatic.to_csv(out_meta, index=False)

print("\n[Done] 已保存：")
print("  Source / Reference:", out_primary)
print("  Target / Query   :", out_meta)
