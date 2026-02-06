import torch
import torch.nn.functional as F

# ======= KDE 工具函数 =======
def _kde_pdf(samples: torch.Tensor, x: torch.Tensor, bw: float) -> torch.Tensor:
    """
    一维高斯核密度估计 (KDE)，纯 torch 实现。
    samples: [N] 数据点
    x: [M] 查询点
    bw: float, 带宽
    return: pdf(x) [M]
    """
    samples = samples.view(-1, 1)     # [N,1]
    x = x.view(1, -1)                 # [1,M]
    z = (samples - x) / bw            # [N,M]
    log_kernel = -0.5 * z.pow(2) - 0.5 * torch.log(
        torch.tensor(2.0 * torch.pi, device=samples.device)
    )
    kern = torch.exp(log_kernel) / bw
    pdf = kern.mean(dim=0)            # [M]
    return pdf


# ======= Overlap Loss =======
def loss_overlap(
    s_u: torch.Tensor,
    s_a: torch.Tensor,
    x_num: int = 1024,
    expand_ratio: float = 0.3,
    bw_min: float = 1e-3,
    device=None
) -> torch.Tensor:
    """
    Overlap loss: 最小化正常 (s_u) 与异常 (s_a) 分布的重叠面积
    s_u: 正常类分数, [Nu]
    s_a: 异常类分数, [Na]
    return: overlap area (scalar tensor)
    """
    if device is None:
        device = s_u.device

    s_u = s_u.view(-1)
    s_a = s_a.view(-1)

    if s_u.numel() < 2 or s_a.numel() < 2:
        # 如果 batch 里类别不足，返回 0 避免报错
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Silverman 规则估计带宽
    n_u = float(s_u.numel())
    n_a = float(s_a.numel())
    bw_u = max((n_u * 3.0 / 4.0) ** (-1.0 / 5.0), bw_min)
    bw_a = max((n_a * 3.0 / 4.0) ** (-1.0 / 5.0), bw_min)

    # 定义积分区间 [xmin, xmax]，外扩 expand_ratio
    xmin = torch.min(torch.min(s_u), torch.min(s_a))
    xmax = torch.max(torch.max(s_u), torch.max(s_a))
    dx = (xmax - xmin) * expand_ratio
    xmin = xmin - dx
    xmax = xmax + dx
    x = torch.linspace(xmin.detach(), xmax.detach(), x_num, device=device)

    # KDE 估计 pdf
    pdf_u = _kde_pdf(s_u, x, bw_u)
    pdf_a = _kde_pdf(s_a, x, bw_a)

    # 交叠面积 = ∫ min(pdf_u, pdf_a) dx
    inter = torch.minimum(pdf_u, pdf_a)
    area = torch.trapz(inter, x)  # scalar

    return area
