import os

import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import Counter
import dgl
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


##  smote采样和下采样相结合 ##
# 定义SMOTE和下采样策略
over = SMOTE(sampling_strategy=0.75)  # 少数类上采样到多数类的50%
under = RandomUnderSampler(sampling_strategy=0.75)  # 多数类下采样到少数类的50%
steps = [('over', over), ('under', under)]
pipeline = Pipeline(steps=steps)

def sample_data(data, sample_rate, random_state=None):
    """随机采样数据集中的样本。
    Args:
        data (DataFrame): 原始数据集。
        sample_rate (float): 采样比例，例如 0.2 表示 20%。
        random_state (int, optional): 随机数种子，用于结果复现。
    Returns:
        DataFrame: 采样后的数据。
    """
    # 计算需要采样的样本数量
    num_samples = int(len(data) * sample_rate)
    # 随机选择样本
    sampled_data = data.sample(n=num_samples, random_state=random_state)
    return sampled_data

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def add_gaussian_noise(X, y, std=0.01, ratio=0.3):
    n_samples = int(len(X) * ratio)
    indices = np.random.choice(len(X), n_samples, replace=False)
    noise = np.random.normal(loc=0.0, scale=std, size=X[indices].shape)
    X_aug = X[indices] + noise
    y_aug = y[indices]
    return np.vstack([X, X_aug]), np.hstack([y, y_aug])


# def load_data_drug(dataset,DRUG,k=15):
#     # print(k)
#     # DRUG = "Gefitinib"
#     data = None
#
#     if dataset == 'source':
#         path_s = './split_norm/' + DRUG + '/' + 'Source_exprs_resp_z.' + DRUG + '.tsv'
#         source_data = pd.read_csv(path_s, sep='\t', index_col=0)
#
#         sample_rate = 1
#         source_data = sample_data(source_data, sample_rate,random_state=42) # 随机采样
#         # print(f'当前的采样率为: {sample_rate * 100}%')
#
#         x_expression = source_data.iloc[:, 2:]  # gene expressions (features)
#         y_logIC50 = source_data.iloc[:, 1]  # col index 1 of the source df is logIC50
#         y_response = source_data.iloc[:, 0]
#         threshold = source_data['logIC50'][source_data['response'] == 0].min()  # 计算 logIC50 列中响应为0的最小值，并将其设置为阈值。
#
#
#         Counter(source_data['response'])[0] / len(source_data['response'])  # 计算 response 列中类别为0的样本占总样本的比例。
#         Counter(source_data['response'])[1] / len(source_data['response'])  # 计算 response 列中类别为1的样本占总样本的比例。
#         class_sample_count_s = np.array([Counter(source_data['response'])[0] / len(source_data['response']),
#                                        Counter(source_data['response'])[1] / len(source_data['response'])])
#         # print(class_sample_count_s)
#
#
#         # 应用SMOTE和下采样
#         x_resampled, y_resampled = pipeline.fit_resample(x_expression, y_response)
#
#         class_counts = Counter(y_resampled)
#         total_samples = len(y_resampled)
#         class_1_proportion = class_counts[1] / total_samples
#
#         print(f"类别为1的样本占总样本的比例: {class_1_proportion:.4f}")
#         print(f"类别为1的样本数: {class_counts[1]}")
#
#         # 标准化处理
#         scaler = StandardScaler()
#         x_scaled = scaler.fit_transform(x_resampled)
#
#         # data （细胞，基因）细胞索引  0..... 基因名
#         data = x_scaled
#         label = y_resampled.values
#         # # 使用 train_test_split 函数分割数据集
#         # X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.4, random_state=42)
#
#
#     elif dataset == 'target':
#         path_s = './split_norm/' + DRUG + '/' + 'Target_expr_resp_z.' + DRUG + '.tsv'
#         target_data = pd.read_csv(path_s, sep='\t', index_col=0)
#         x_expression = target_data.iloc[:, 1:]  # gene expressions (features)
#         y_response = target_data.iloc[:, 0]  # col index 1 of the source df is logIC50
#         data = x_expression
#         label = y_response.values
#
#     scaled = data
#
#     # 将数据转换为numpy矩阵
#     matrix = scaled.to_numpy() if isinstance(scaled, pd.DataFrame) else scaled
#     features = torch.from_numpy(matrix).float()
#     # 生成k近邻图的稀疏矩阵表示
#     # knn_adj = kneighbors_graph(features, n_neighbors=k, mode='distance', include_self=True,
#     #                                metric='cosine')
#     knn_adj = dgl.knn_graph(features, k=k, algorithm='kd-tree', dist='cosine') ## 使用DGL库生成k近邻图
#     # 修改部分：确保返回的adj是DGLGraph对象
#     g = dgl.add_self_loop(knn_adj) # 为图添加自环
#     # adj = g.adjacency_matrix_scipy(return_edge_ids=False).tocoo()
#     adj = g.adjacency_matrix()  # 获取邻接矩阵
#     row, col= adj.coo()   ## 将邻接矩阵转换为COO格式
#     row = row.numpy()
#     col = col.numpy()
#     coo_adj_data = adj.val.numpy()    # 获取COO格式的邻接矩阵数据
#     # 创建 SciPy 的 COO 稀疏矩阵
#     sp_adj = sp.coo_matrix((coo_adj_data, (row, col)), shape=(g.number_of_nodes(), g.number_of_nodes()))
#     adj_normalized = normalize_adj(sp_adj)  # 归一化邻接矩阵
#
#     labels = torch.LongTensor(label)
#     # adj = sparse_mx_to_torch_sparse_tensor(adj)
#     # 返回特征数量
#     n_features = features.shape[1]
#
#     # 检查输入数据是否存在 NaN
#     if np.isnan(features.numpy()).any():
#         print("NaN detected in features")
#     if np.isnan(labels.numpy()).any():
#         print("NaN detected in labels")
#
#     return adj_normalized, features, labels, knn_adj, n_features

def load_data_cellStatus(dataset, DRUG, k):
    assert dataset in ['source', 'target']

    # if dataset == 'source':
    #     path_s = '../CSdata8/' + DRUG + '/' + 'Source_exprs_with_label_' + DRUG + '.tsv'
    #     df_s = pd.read_csv(path_s, sep='\t')
    #
    #     feature_df = df_s.iloc[:, 3:-2]  # 跳过前3列：CellID, response, logIC50；最后两列是PseudoLabel和LabelID
    #     labels = df_s['LabelID'].values
    #
    # else:
    #     path_t = '../CSdata8/' + DRUG + '/' + 'Target_exprs_with_label_' + DRUG + '.tsv'
    #     df_t = pd.read_csv(path_t, sep='\t')
    #     feature_df = df_t.iloc[:, 2:-2]  # 跳过前2列：CellID, response；最后两列是PseudoLabel和LabelID
    #     labels = df_t['LabelID'].values

    if dataset == 'source':
        path = f"./Cross_Platform/Sc_matrix.{DRUG}_GSE108383_451Lu_withPseudoNum.csv"
    else:
        path = f"./Cross_Platform/Sc_matrix.{DRUG}_GSE108394_451Lu_withPseudoNum.csv"

    # 读取数据
    df = pd.read_csv(path, index_col=0)
    feature_df = df.drop(columns=["response", "pseudo_label"]).values
    labels = df["pseudo_label"].values

    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_df)

    # 转为 torch tensor
    features = torch.FloatTensor(features_scaled)
    labels = torch.LongTensor(labels)

    # 构建 kNN 图（用 DGL）
    knn_graph = dgl.knn_graph(features, k=k, algorithm='kd-tree', dist='cosine')
    knn_graph = dgl.add_self_loop(knn_graph)

    # 获取 scipy 稀疏邻接矩阵（用于 Pro-GNN）
    adj = knn_graph.adjacency_matrix().to_dense().numpy()
    row, col = np.where(adj > 0)
    data = adj[row, col]
    sp_adj = sp.coo_matrix((data, (row, col)), shape=(features.shape[0], features.shape[0]))
    adj_normalized = normalize_adj(sp_adj)

    return adj_normalized, features, labels, knn_graph, features.shape[1]

# def load_data_drug_from_prognn(dataset, DRUG):
#     assert dataset in ['source', 'target']
#
#     # 1. 读 ProGNN 学到的图
#     # npz_path = f'../graph_8status/clean_graph_{dataset}_{DRUG}.npz'
#     if dataset == 'source':
#         npz_path = "../scToscData/Cross_Platform/source_PLX4720_GSE108383_A375_withPseudoNum.npz"
#     else:
#         npz_path = "../scToscData/Cross_Platform/target_PLX4720_GSE108394_451Lu_withPseudoNum.npz"
#
#     sp_adj = sp.load_npz(npz_path)        # scipy.sparse.coo_matrix
#     sp_adj = sp_adj.tocoo()
#     src, dst = sp_adj.row, sp_adj.col
#     g = dgl.graph((src, dst), num_nodes=sp_adj.shape[0])
#     g = dgl.add_self_loop(g)
#
#     # 2. 读特征 & 标签
#
#     # if dataset == 'source':
#     #     path = f'../CSdata8/{DRUG}/Source_exprs_with_label_{DRUG}.tsv'
#     #     df = pd.read_csv(path, sep='\t')
#     #     feature_df = df.iloc[:, 3:-2]
#     #     labels = df['response'].values
#     # else:
#     #     path = f'../CSdata8/{DRUG}/Target_exprs_with_label_{DRUG}.tsv'
#     #     df = pd.read_csv(path, sep='\t')
#     #     feature_df = df.iloc[:, 2:-2]     # 跳过 CellID,response
#     #     labels = df['response'].values
#
#     if dataset == 'source':
#         path = "../scToscData/Cross_CellLine/Processed_GSE108383_451Lu_withPseudoNum.csv"
#     else:
#         path = "../scToscData/Cross_CellLine/Processed_GSE108383_A375_withPseudoNum.csv"
#     df = pd.read_csv(path, index_col=0)
#     feature_df = df.drop(columns=["response", "pseudo_label"])  # 只保留基因表达
#     labels = df["response"].values  # 用伪标签
#
#     # 3. 标准化特征
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(feature_df)
#     features = torch.FloatTensor(features_scaled)
#     labels = torch.LongTensor(labels)
#
#     n_features = features.shape[1]
#
#     return g, features, labels, n_features

# def load_data_drug_from_prognn(dataset, DRUG):
#     """
#     从 scToscData/1~6 中加载 ProGNN 生成的图 + 特征 + 标签
#
#     参数
#     ----
#     dataset : 'source' 或 'target'
#     DRUG    : 药物名称（大小写不敏感），例如：
#               'PLX4720', 'paclitaxel+atezolizumab',
#               'paclitaxel', 'cisplatin', 'erlotinib',
#               或者扩展版 'plx4720_a375'
#     返回
#     ----
#     g           : DGLGraph（已经加了 self-loop）
#     features    : torch.FloatTensor, 形状 [N, F]
#     labels      : torch.LongTensor,   形状 [N]
#     n_features  : 特征维度 F
#     """
#
#     assert dataset in ['source', 'target']
#     drug_key = DRUG.lower()
#
#     # 根据你之前给的 1~6 目录结构，做一个配置表
#     drug_config = {
#         # 1号文件夹：PLX4720，451Lu vs 451Lu（GSE108383 -> GSE108394）
#         'plx4720': {
#             'folder': '1',
#             'source_csv_kw': 'GSE108383_451Lu',
#             'target_csv_kw': 'GSE108394_451Lu',
#         },
#         # 2号文件夹：PLX4720，451Lu -> A375（都是 GSE108383）
#         'plx4720_a375': {
#             'folder': '2',
#             'source_csv_kw': 'GSE108383_451Lu',
#             'target_csv_kw': 'GSE108383_A375',
#         },
#         # 3号文件夹：Paclitaxel + Atezolizumab，Tissue -> Blood
#         'paclitaxel+atezolizumab': {
#             'folder': '3',
#             'source_csv_kw': 'PacAteTissue',
#             'target_csv_kw': 'PacAteBlood',
#         },
#         # 4号文件夹：Paclitaxel，Tissue -> Blood
#         'paclitaxel': {
#             'folder': '4',
#             'source_csv_kw': 'PacTissue',
#             'target_csv_kw': 'PacBlood',
#         },
#         # 5号文件夹：Cisplatin，Primary -> Metastatic
#         'cisplatin': {
#             'folder': '5',
#             'source_csv_kw': 'Primary',
#             'target_csv_kw': 'Metastatic',
#         },
#         # 6号文件夹：Erlotinib，bulk -> sc
#         'erlotinib': {
#             'folder': '6',
#             'source_csv_kw': 'bulk',
#             'target_csv_kw': 'sc',
#         },
#     }
#
#     if drug_key not in drug_config:
#         raise ValueError(f"不支持的 DRUG：{DRUG}，请检查 drug_config 映射。")
#
#     cfg = drug_config[drug_key]
#     base_dir = os.path.join('../scToscData', cfg['folder'])
#     if not os.path.isdir(base_dir):
#         raise FileNotFoundError(f"目录不存在：{base_dir}")
#
#     files = os.listdir(base_dir)
#
#     # ---------- 1. 选择图文件（.npz） ----------
#     if dataset == 'source':
#         npz_candidates = [
#             f for f in files
#             if f.endswith('.npz') and 'clean_graph_source' in f
#         ]
#     else:
#         npz_candidates = [
#             f for f in files
#             if f.endswith('.npz') and 'clean_graph_target' in f
#         ]
#
#     if len(npz_candidates) == 0:
#         raise FileNotFoundError(f"{base_dir} 中没有找到对应的 npz 图文件（{dataset}）")
#
#     if len(npz_candidates) > 1:
#         print(f"[警告] 在 {base_dir} 中找到多个 npz 候选，默认使用第一个：{npz_candidates}")
#
#     npz_path = os.path.join(base_dir, npz_candidates[0])
#
#     sp_adj = sp.load_npz(npz_path).tocoo()
#     src, dst = sp_adj.row, sp_adj.col
#     g = dgl.graph((src, dst), num_nodes=sp_adj.shape[0])
#     g = dgl.add_self_loop(g)
#
#     # ---------- 2. 选择特征 + 标签的 CSV ----------
#     if dataset == 'source':
#         csv_kw = cfg['source_csv_kw']
#     else:
#         csv_kw = cfg['target_csv_kw']
#
#     csv_candidates = [
#         f for f in files
#         if f.endswith('.csv') and csv_kw in f
#     ]
#
#     if len(csv_candidates) == 0:
#         raise FileNotFoundError(
#             f"{base_dir} 中没有找到包含关键字 '{csv_kw}' 的 CSV 文件（{dataset}）"
#         )
#
#     if len(csv_candidates) > 1:
#         print(f"[警告] 在 {base_dir} 中找到多个 CSV 候选，默认使用第一个：{csv_candidates}")
#
#     csv_path = os.path.join(base_dir, csv_candidates[0])
#
#     df = pd.read_csv(csv_path, index_col=0)
#
#     # 这里假设列里包含 'response' 和 'pseudo_label'
#     # 如果你想用 pseudo_label 做标签，把下面那行改成 df['pseudo_label']
#     if 'response' not in df.columns or 'pseudo_label' not in df.columns:
#         raise ValueError(f"{csv_path} 中没有找到 'response' 或 'pseudo_label' 列，请检查列名。")
#
#     feature_df = df.drop(columns=['response', 'pseudo_label'])
#     labels_np = df['response'].values.astype(int)
#
#     # ---------- 3. 标准化特征 ----------
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(feature_df.values)
#
#     features = torch.FloatTensor(features_scaled)
#     labels = torch.LongTensor(labels_np)
#     n_features = features.shape[1]
#
#     # 简单 sanity check
#     if np.isnan(features.numpy()).any():
#         print(f"[警告] 特征中存在 NaN：{csv_path}")
#     if np.isnan(labels.numpy()).any():
#         print(f"[警告] 标签中存在 NaN：{csv_path}")
#
#     return g, features, labels, n_features


def load_data_drug_from_prognn(dataset, DRUG):
    """
    从 scToscData/1~6 中加载 ProGNN 生成的图 + 特征 + 标签

    参数
    ----
    dataset : 'source' 或 'target'
    DRUG    : 药物名称（大小写不敏感），例如：
              'PLX4720', 'paclitaxel+atezolizumab',
              'paclitaxel', 'cisplatin', 'erlotinib',
              或者扩展版 'plx4720_a375'
    返回
    ----
    g           : DGLGraph（已经加了 self-loop）
    features    : torch.FloatTensor, 形状 [N, F]
    labels      : torch.LongTensor,   形状 [N]
    n_features  : 特征维度 F
    """

    assert dataset in ['source', 'target']
    drug_key = DRUG.lower()

    drug_config = {
        'plx4720': {
            'folder': '1',
            'source_csv_kw': 'GSE108383_451Lu',
            'target_csv_kw': 'GSE108394_451Lu',
        },
        'plx4720_a375': {
            'folder': '2',
            'source_csv_kw': 'GSE108383_451Lu',
            'target_csv_kw': 'GSE108383_A375',
        },
        'paclitaxel+atezolizumab': {
            'folder': '3',
            'source_csv_kw': 'PacAteTissue',
            'target_csv_kw': 'PacAteBlood',
        },
        'paclitaxel': {
            'folder': '4',
            'source_csv_kw': 'PacTissue',
            'target_csv_kw': 'PacBlood',
        },
        'cisplatin': {
            'folder': '5',
            'source_csv_kw': 'Primary',
            'target_csv_kw': 'Metastatic',
        },
    }

    if drug_key not in drug_config:
        raise ValueError(f"不支持的 DRUG：{DRUG}，请检查 drug_config 映射。")

    cfg = drug_config[drug_key]
    base_dir = os.path.join('../scToscData', cfg['folder'])
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"目录不存在：{base_dir}")

    files = os.listdir(base_dir)

    # ---------- 1. 选择图文件（.npz） ----------
    if dataset == 'source':
        npz_candidates = [
            f for f in files
            if f.endswith('.npz') and 'clean_graph_source' in f
        ]
    else:
        npz_candidates = [
            f for f in files
            if f.endswith('.npz') and 'clean_graph_target' in f
        ]

    if len(npz_candidates) == 0:
        raise FileNotFoundError(f"{base_dir} 中没有找到对应的 npz 图文件（{dataset}）")

    if len(npz_candidates) > 1:
        print(f"[警告] 在 {base_dir} 中找到多个 npz 候选，默认使用第一个：{npz_candidates}")

    npz_path = os.path.join(base_dir, npz_candidates[0])

    sp_adj = sp.load_npz(npz_path).tocoo()
    src, dst = sp_adj.row, sp_adj.col
    g = dgl.graph((src, dst), num_nodes=sp_adj.shape[0])
    g = dgl.add_self_loop(g)

    # ---------- 2. 选择特征 + 标签的 CSV ----------
    if dataset == 'source':
        csv_kw = cfg['source_csv_kw']
    else:
        csv_kw = cfg['target_csv_kw']

    csv_candidates = [
        f for f in files
        if f.endswith('.csv') and csv_kw in f
    ]

    if len(csv_candidates) == 0:
        raise FileNotFoundError(
            f"{base_dir} 中没有找到包含关键字 '{csv_kw}' 的 CSV 文件（{dataset}）"
        )

    if len(csv_candidates) > 1:
        print(f"[警告] 在 {base_dir} 中找到多个 CSV 候选，默认使用第一个：{csv_candidates}")

    csv_path = os.path.join(base_dir, csv_candidates[0])

    df = pd.read_csv(csv_path, index_col=0)

    # 这里假设列里包含 'response' 和 'pseudo_label'
    # 如果你想用 pseudo_label 做标签，把下面那行改成 df['pseudo_label']
    if 'response' not in df.columns or 'pseudo_label' not in df.columns:
        raise ValueError(f"{csv_path} 中没有找到 'response' 或 'pseudo_label' 列，请检查列名。")

    feature_df = df.drop(columns=['response', 'pseudo_label'])
    labels_np = df['response'].values.astype(int)

    # ---------- 3. 标准化特征 ----------
    # 针对数据集 3 和 4 判断是否需要归一化
    if DRUG.lower() in ['paclitaxel', 'paclitaxel+atezolizumab']:
        feature_mean = feature_df.mean(axis=0)
        feature_std = feature_df.std(axis=0)

        # 检查是否已经归一化（均值接近0，标准差接近1）
        if np.allclose(feature_mean, 0, atol=1e-3) and np.allclose(feature_std, 1, atol=1e-3):
            print(f"[信息] 数据集 {DRUG} 已经归一化，跳过归一化。")
            features = torch.FloatTensor(feature_df.values)  # 不进行归一化
        else:
            # 进行标准化
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(feature_df.values)
            features = torch.FloatTensor(features_scaled)
    else:
        # 对其他数据集进行标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_df.values)
        features = torch.FloatTensor(features_scaled)


    labels = torch.LongTensor(labels_np)  # 转为 PyTorch 张量

    n_features = features.shape[1]

    # 简单 sanity check
    if np.isnan(features.numpy()).any():
        print(f"[警告] 特征中存在 NaN：{csv_path}")
    if np.isnan(labels.numpy()).any():
        print(f"[警告] 标签中存在 NaN：{csv_path}")

    return g, features, labels, n_features

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)


# def load_adj_label_drug(graph):
#     # 提取邻接矩阵的边
#     src, dst = graph.edges()
#     # 创建一个SciPy稀疏矩阵
#     A = sp.coo_matrix((np.ones(len(src)), (src.numpy(), dst.numpy())), shape=(graph.number_of_nodes(), graph.number_of_nodes()))
#     adj_label = torch.FloatTensor(A.toarray())
#     pos_weight = float(A.shape[0] * A.shape[0] - A.sum()) / A.sum()
#     pos_weight = np.array(pos_weight).reshape(1, 1)
#     pos_weight = torch.from_numpy(pos_weight)
#     norm = A.shape[0] * A.shape[0] / float((A.shape[0] * A.shape[0] - A.sum()) * 2)
#     return adj_label, pos_weight, norm


# def compute_positional_encodings(graph, num_eigenvectors):
#     """
#     计算给定DGL图的Laplacian位置编码。
#
#     参数:
#     - graph: DGLGraph，输入的图。
#     - num_eigenvectors: int，要计算的最小非平凡特征向量的数量。
#
#     返回:
#     - Tensor，形状为(N, k)，N是节点数，k是特征向量的数量。
#     """
#     # 使用dgl.lap_pe计算位置编码
#     pos_enc = dgl.lap_pe(graph, k=num_eigenvectors, padding=True)
#     # 标准化位置编码
#     pos_enc_mean = pos_enc.mean(dim=0, keepdim=True)
#     pos_enc_std = pos_enc.std(dim=0, keepdim=True)
#     pos_enc_normalized = (pos_enc - pos_enc_mean) / (pos_enc_std + 1e-6)  # 添加小常数避免除以零
#     return pos_enc_normalized