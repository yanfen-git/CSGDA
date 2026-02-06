# coding=utf-8
import os
import torch.backends.cudnn as cudnn
from GT_drug import GT
from overlapLoss import loss_overlap
from preprocess_smote import *
import torch
from torch import nn
import torch.nn.functional as F
import itertools
import random
import torch.utils.data
import argparse
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import confusion_matrix


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--cuda', type=str, default="0")
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--hidden', type=int, default=1024,
                    help='Number of hidden units.')
parser.add_argument('--gfeat', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--nfeat', type=float, default=2000,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--classes', type=int, default=2,
                    help='classes number')
parser.add_argument('--model_path', type=str, default='models')
parser.add_argument('--lambda_d', type=float, default=0,
                    help='hyperparameter for domain loss')
parser.add_argument('--drug_name','-d',help='drug name')
parser.add_argument('--lambda_overlap', type=float, default=1,
                    help='weight for overlap loss on source')
parser.add_argument('--lambda_ce', type=float, default=1,
                    help='optional CE weight (0 = disable)')

# The --drug_name passed at runtime must exist in drug_config. Examples:
#
# python xx.py -d PLX4720
#
# python xx.py -d paclitaxel
#
# python xx.py -d paclitaxel+atezolizumab
#
# python xx.py -d cisplatin
#
# To use the 451Lu -> A375 task from folder #2, use:
# python xx.py -d plx4720_a375

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
DRUG = args.drug_name
cuda = True
cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
manual_seed = random.randint(1, 10000)
print("manual_seed:", manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)


''' Load data '''
adj_s, features_s, labels_s, n_features_s = load_data_drug_from_prognn('source', DRUG)
adj_t, features_t, labels_t, n_features_t = load_data_drug_from_prognn('target', DRUG)


domain_labels_s = torch.zeros(features_s.shape[0], dtype=torch.long).to(device)
domain_labels_t = torch.ones(features_t.shape[0], dtype=torch.long).to(device)


adj_s = adj_s.to(device)
features_s = features_s.to(device)
labels_s = labels_s.to(device)

adj_t = adj_t.to(device)
features_t = features_t.to(device)
labels_t = labels_t.to(device)


adj_s.ndata['feat'] = features_s
adj_s.ndata['label'] = labels_s
adj_t.ndata['feat'] = features_t
adj_t.ndata['label'] = labels_t


args.nfeat = n_features_s

def predict(adj, feature):
    z = shared_encoder(adj, feature)
    logits = cls_model(z)
    if torch.isnan(logits).any():
        print("NaN detected in logits")
    return logits

def evaluate(preds, labels):
    accuracy1 = accuracy(preds, labels)
    return accuracy1


def test(adj,feature, label):  # def test(feature, adj, ppmi, label):
    for model in models:
        model.eval()
    logits = predict(adj,feature)   # logits = predict(feature, adj, ppmi)
    labels = label
    accuracy = evaluate(logits, labels)

    logits_proba = F.softmax(logits, dim=1).cpu().detach().numpy()
    preds = logits_proba.argmax(axis=1)  # 获取预测的标签
    labels_np = labels.cpu().detach().numpy()


    if np.isnan(logits_proba).any():
        print("NaN detected in logits_proba")
    if np.isnan(labels_np).any():
        print("NaN detected in labels_np")


    cm = confusion_matrix(labels_np, preds)
    tn, fp, fn, tp = cm.ravel()


    auc = roc_auc_score(labels_np, logits_proba[:, 1])


    aupr = average_precision_score(labels_np, logits_proba[:, 1])

    f1m = f1_score(labels_np, preds, average='macro')
    return accuracy, auc, aupr, f1m

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * rate
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)



# 检查数据是否包含 NaN
def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")

''' set loss function '''
ce_loss_fn = nn.CrossEntropyLoss().to(device)
domain_loss_fn = nn.NLLLoss().to(device)



''' load model '''

''' shared encoder (including Local GCN and Global GCN) '''
shared_encoder = GT(nfeat=args.nfeat, nhid=args.hidden, nclass=args.gfeat, dropout=args.dropout).to(device)



''' node classifier model '''
cls_model = nn.Sequential(
    nn.Linear(args.gfeat, 2),
).to(device)

''' domain discriminator model '''

domain_model = nn.Sequential(
    GRL(),
    nn.Linear(args.gfeat, 20),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(20, 2),
).to(device)



''' the set of models used in ASN '''
models = [ shared_encoder, cls_model, domain_model]
params = itertools.chain(*[model.parameters() for model in models])

''' setup optimizer '''
optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=5e-4)


def save_model(model, filename):
    torch.save(model.state_dict(), filename)

''' training '''
best_acc_s = 0
best_auc_s = 0
best_aupr_s = 0
best_f1_s = 0
best_acc_t = 0
best_auc_t = 0
best_aupr_t = 0
best_f1_t = 0
best_val_loss = float('inf')
acc_t_list = []
auc_t_list = []
aupr_t_list = []

losses=[]

for epoch in range(args.n_epoch):
    embedding_dir = f"./embedding/{args.drug_name}/epoch_{epoch}"
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)


    len_dataloader = min(labels_s.shape[0], labels_t.shape[0])
    global rate
    rate = min((epoch + 1) / args.n_epoch, 0.05)

    for model in models:
        model.train()
    optimizer.zero_grad()

    ''' Share encoder '''
    z_s = shared_encoder(adj_s, features_s)
    z_t = shared_encoder(adj_t, features_t)


    # ''' compute node classification loss for S '''
    # source_logits = cls_model(z_s)
    # cls_loss_source = cls_loss(source_logits, labels_s)

    ''' compute node classification loss for S (Overlap loss) '''
    source_logits = cls_model(z_s)

    # 得到二分类概率分数
    probs = F.softmax(source_logits, dim=1)
    s_u = probs[labels_s == 0, 1]
    s_a = probs[labels_s == 1, 1]

    if s_u.numel() > 1 and s_a.numel() > 1:
        overlap_loss_val = loss_overlap(s_u, s_a, device=device)
        ce_loss_val = ce_loss_fn(source_logits, labels_s)
        cls_loss_source = args.lambda_overlap * overlap_loss_val + args.lambda_ce * ce_loss_val
    else:
        # batch里只有单类，退回CE
        cls_loss_source = ce_loss_fn(source_logits, labels_s)

    ''' compute domain classifier loss for both S and T '''
    domain_output_s = domain_model(z_s)
    domain_output_t = domain_model(z_t)
    err_s_domain = ce_loss_fn(domain_output_s,
                              torch.zeros(domain_output_s.size(0)).long().to(device))
    err_t_domain = ce_loss_fn(domain_output_t,
                              torch.ones(domain_output_t.size(0)).long().to(device))
    loss_grl = err_s_domain + err_t_domain

    # warm-up: 前 30% epoch 逐渐拉升 λ_d
    lambda_d_curr = args.lambda_d * min((epoch + 1) / (args.n_epoch * 0.3), 0.6)

    ''' compute entropy loss for T '''
    target_logits = cls_model(z_t)
    target_probs = F.softmax(target_logits, dim=-1)
    loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs + 1e-12), dim=-1))

    # 动态权重
    lambda_ent = 0.1 * (epoch / args.n_epoch)

    ''' compute overall loss '''
    loss = cls_loss_source + lambda_d_curr * loss_grl + lambda_ent * loss_entropy


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # # （可选）仅保存共享嵌入 z，用于可视化
    # if epoch == args.n_epoch - 1 or True:
    #     save_dir = f"./pri_share_embed/{DRUG}"
    #     os.makedirs(save_dir, exist_ok=True)
    #     z_s_np = z_s.detach().cpu().numpy()
    #     z_t_np = z_t.detach().cpu().numpy()
    #     df_shared = pd.DataFrame(np.vstack([z_s_np, z_t_np]))
    #     df_shared['domain'] = np.concatenate([domain_labels_s.cpu().numpy(),
    #                                           domain_labels_t.cpu().numpy()])
    #     df_shared.to_csv(f"{save_dir}/epoch_{epoch}_shared_domain.csv", index=False)

    if (epoch + 1) % 1 == 0:

        acc_s, auc_s, aupr_s, f1_s = test(adj_s, features_s, labels_s)
        if acc_s > best_acc_s:
            best_acc_s = acc_s
        if auc_s >best_auc_s:
            best_auc_s = auc_s
        if aupr_s > best_aupr_s:
            best_aupr_s = aupr_s
        if f1_s > best_f1_s:
            best_f1_s = f1_s
        acc_t, auc_t, aupr_t, f1_t = test(adj_t, features_t, labels_t)
        acc_t_list.append(acc_t.item())
        auc_t_list.append(auc_t)
        aupr_t_list.append(aupr_t)
        if acc_t > best_acc_t:
            best_acc_t = acc_t
        if auc_t > best_auc_t:
            best_auc_t = auc_t
        if aupr_t > best_aupr_t:
            best_aupr_t = aupr_t
        if f1_t > best_f1_t:
            best_f1_t = f1_t
        print(
            f"epoch: {epoch}, "
            f"acc_source: {acc_s:.4f}, AUC_source: {auc_s:.4f}, AUPR_source: {aupr_s:.4f}, F1_source(macro): {f1_s:.4f}, "
            f"acc_target: {acc_t:.4f}, AUC_target: {auc_t:.4f}, AUPR_target: {aupr_t:.4f}, F1_target(macro): {f1_t:.4f}, "
            f"loss_class: {cls_loss_source.item():.6f}, "
            f"loss_domain: {loss_grl.item():.6f} (λ_d={lambda_d_curr:.3f}), "
            f"loss_entropy: {loss_entropy.item():.6f} (λ_ent={lambda_ent:.3f})"
        )

print("=============================================================")
print(f"Best results on TARGET domain:")
print(f"  ACC : {best_acc_t:.4f}")
print(f"  AUC : {best_auc_t:.4f}")
print(f"  AUPR: {best_aupr_t:.4f}")
print(f"  F1  : {best_f1_t:.4f}")

# import pandas as pd
# pd.DataFrame(losses).to_csv("loss.csv")

