
import os
import argparse
import numpy as np
import torch
import scipy.sparse as sp

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from deeprobust.graph.defense import GCN, ProGNN
from deeprobust.graph.utils import preprocess

from preprocess import load_data_cellStatus


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--debug", action="store_true", default=False,
                        help="debug mode for ProGNN")

    parser.add_argument(
        "--drug",
        type=str,
        default="PLX4720",
        help="drug name"
    )

    parser.add_argument("--hidden", type=int, default=16)

    # GCN / ProGNN 相关超参（你之前日志里的设置）
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--epochs", type=int, default=300)

    parser.add_argument("--alpha", type=float, default=5e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambda_", type=float, default=0.1)
    parser.add_argument("--phi", type=float, default=0.0)

    parser.add_argument("--inner_steps", type=int, default=2)
    parser.add_argument("--outer_steps", type=int, default=1)
    parser.add_argument("--lr_adj", type=float, default=0.01)

    parser.add_argument("--symmetric", action="store_true", default=False)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only_gcn", action="store_true", default=False)
    parser.add_argument("--cuda", type=str, default="0")

    # Erlotinib 这组我用 k_s=15, k_t=10, val_ratio=0.2
    parser.add_argument("--k_source", type=int, default=15)
    parser.add_argument("--k_target", type=int, default=10)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    return parser.parse_args()


def get_device(args):
    if args.cuda.lower() == "cpu" or args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.cuda}")
        torch.cuda.set_device(int(args.cuda))
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return device


def safe_stratified_split(idx, labels, val_ratio=0.2, random_state=42):
    """
    默认按标签分层划分，如果某个标签太少导致报错，就退回普通随机划分。
    """
    try:
        idx_train, idx_val = train_test_split(
            idx,
            test_size=val_ratio,
            stratify=labels,
            random_state=random_state,
        )
        print("[info] stratified split OK")
        return idx_train, idx_val
    except ValueError as e:
        print(f"[WARN] stratified split 失败: {e}")
        print("[WARN] 改用不带 stratify 的随机划分。")
        idx_train, idx_val = train_test_split(
            idx,
            test_size=val_ratio,
            random_state=random_state,
        )
        return idx_train, idx_val


def save_tensor_as_npz(tensor: torch.Tensor, path: str):
    """
    将 torch.Tensor (稀疏或稠密) 保存为 SciPy COO npz 文件。
    """
    if isinstance(tensor, torch.Tensor) and tensor.is_sparse:
        tensor = tensor.coalesce()
        indices = tensor.indices().cpu().numpy()
        values = tensor.values().cpu().numpy()
        row, col = indices
        sp_tensor = sp.coo_matrix((values, (row, col)), shape=tensor.shape)
    else:
        array = tensor.cpu().detach().numpy()
        sp_tensor = sp.coo_matrix(array)

    sp.save_npz(path, sp_tensor)


# -----------------------------
#  主流程
# -----------------------------
def main():
    args = parse_args()
    device = get_device(args)
    drug = args.drug.lower()

    k_s = args.k_source
    k_t = args.k_target
    val_ratio = args.val_ratio

    print(
        f"\n===> Generating clean graphs for drug: {args.drug} "
        f"(k_s={k_s}, k_t={k_t}, val_ratio={val_ratio})"
    )
    print(
        f"[INFO] hyper-params: hidden={args.hidden}, dropout={args.dropout}, "
        f"alpha={args.alpha}, beta={args.beta}, lambda={args.lambda_}, "
        f"phi={args.phi}, symmetric={args.symmetric}"
    )

    # ----------------------------
    # 1) 加载 source / target 数据
    #    load_data_cellStatus 已经在 preprocess_smote.py 里实现了 erlotinib
    # ----------------------------
    adj_s, features_s, labels_s, _, _ = load_data_cellStatus("source", args.drug, k=k_s)
    adj_t, features_t, labels_t, _, _ = load_data_cellStatus("target", args.drug, k=k_t)

    # ----------------------------
    # 2) Source ProGNN
    # ----------------------------
    idx_s = np.arange(features_s.shape[0])
    idx_s_train, idx_s_val = safe_stratified_split(
        idx_s, labels_s.numpy(), val_ratio=val_ratio, random_state=args.seed
    )

    adj_s_norm, features_s_norm, labels_s = preprocess(
        adj_s, features_s, labels_s, preprocess_adj=False, device=device
    )

    model_s = GCN(
        nfeat=features_s_norm.shape[1],
        nhid=args.hidden,
        nclass=labels_s.max().item() + 1,
        dropout=args.dropout,
        device=device,
    )

    if args.only_gcn:
        prognn_s = model_s
        prognn_s.fit(features_s_norm, adj_s_norm, labels_s, idx_s_train, idx_s_val)
        best_graph_s = adj_s_norm
        best_val_acc_s = getattr(prognn_s, "best_val_acc", 0.0)
    else:
        prognn_s = ProGNN(model_s, args, device)
        prognn_s.fit(features_s_norm, adj_s_norm, labels_s, idx_s_train, idx_s_val)

        if hasattr(prognn_s, "best_graph"):
            best_graph_s = prognn_s.best_graph
        else:
            print("[WARN] prognn_s 没有 best_graph 属性，退回使用 adj_s_norm")
            best_graph_s = adj_s_norm

        best_val_acc_s = getattr(prognn_s, "best_val_acc", 0.0)

    # ------ 手动检查 source 验证集性能 ------
    print("\n[MANUAL CHECK - SOURCE]")
    prognn_s.model.eval()
    with torch.no_grad():
        out_s = prognn_s.model(features_s_norm, adj_s_norm)
        preds_s = out_s.max(1)[1].cpu().numpy()
    labels_s_np = labels_s.cpu().numpy()
    val_preds_s = preds_s[idx_s_val]
    val_labels_s = labels_s_np[idx_s_val]
    acc_manual_s = accuracy_score(val_labels_s, val_preds_s)
    cm_s = confusion_matrix(val_labels_s, val_preds_s)
    print("manual val acc (source) =", acc_manual_s)
    print("confusion matrix (source):\n", cm_s)
    print("reported best_val_acc (source) =", best_val_acc_s)

    # ----------------------------
    # 3) Target ProGNN
    # ----------------------------
    idx_t = np.arange(features_t.shape[0])
    idx_t_train, idx_t_val = safe_stratified_split(
        idx_t, labels_t.numpy(), val_ratio=val_ratio, random_state=args.seed
    )

    adj_t_norm, features_t_norm, labels_t = preprocess(
        adj_t, features_t, labels_t, preprocess_adj=False, device=device
    )

    model_t = GCN(
        nfeat=features_t_norm.shape[1],
        nhid=args.hidden,
        nclass=labels_t.max().item() + 1,
        dropout=args.dropout,
        device=device,
    )

    if args.only_gcn:
        prognn_t = model_t
        prognn_t.fit(features_t_norm, adj_t_norm, labels_t, idx_t_train, idx_t_val)
        best_graph_t = adj_t_norm
        best_val_acc_t = getattr(prognn_t, "best_val_acc", 0.0)
    else:
        prognn_t = ProGNN(model_t, args, device)
        prognn_t.fit(features_t_norm, adj_t_norm, labels_t, idx_t_train, idx_t_val)

        if hasattr(prognn_t, "best_graph"):
            best_graph_t = prognn_t.best_graph
        else:
            print("[WARN] prognn_t 没有 best_graph 属性，退回使用 adj_t_norm")
            best_graph_t = adj_t_norm

        best_val_acc_t = getattr(prognn_t, "best_val_acc", 0.0)

    # ------ 手动检查 target 验证集性能（关键） ------
    print("\n[MANUAL CHECK - TARGET]")
    prognn_t.model.eval()
    with torch.no_grad():
        out_t = prognn_t.model(features_t_norm, adj_t_norm)
        preds_t = out_t.max(1)[1].cpu().numpy()
    labels_t_np = labels_t.cpu().numpy()
    val_preds_t = preds_t[idx_t_val]
    val_labels_t = labels_t_np[idx_t_val]
    acc_manual_t = accuracy_score(val_labels_t, val_preds_t)
    cm_t = confusion_matrix(val_labels_t, val_preds_t)
    print("manual val acc (target) =", acc_manual_t)
    print("confusion matrix (target):\n", cm_t)
    print("reported best_val_acc (target) =", best_val_acc_t)

    # ----------------------------
    # 4) 保存 learned clean graphs
    # ----------------------------
    out_dir = "../graph_8status/new/3"
    os.makedirs(out_dir, exist_ok=True)

    src_path = os.path.join(out_dir, f"clean_graph_source_{args.drug}.npz")
    tgt_path = os.path.join(out_dir, f"clean_graph_target_{args.drug}.npz")

    save_tensor_as_npz(best_graph_s, src_path)
    save_tensor_as_npz(best_graph_t, tgt_path)

    print(f"\n[✓] Saved clean source graph: {src_path}")
    print(f"[✓] Saved clean target graph: {tgt_path}")
    print(f"[source] Best acc_val (from ProGNN): {best_val_acc_s:.4f}")
    print(f"[target] Best acc_val (from ProGNN): {best_val_acc_t:.4f}")
    print(f"[target] Manual val acc (GCN on best graph): {acc_manual_t:.4f}")


if __name__ == "__main__":
    main()
