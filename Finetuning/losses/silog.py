import torch
import torch.nn as nn

class SiLogLoss(nn.Module):
    """ Scale-invariant log RMSE（SigLoss）。 """
    def __init__(self, lam: float = 0.85, eps: float = 1e-6):
        super().__init__()
        self.lam = lam
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor):
        # pred/target: [B,1,H,W] (m), valid: [B,1,H,W] (1=valid)
        mask = (valid > 0.5)
        if mask.sum() == 0:  # 安全策
            return pred.new_tensor(0.0, requires_grad=True)
        
        p = torch.log(torch.clamp(pred[mask], min=self.eps))
        t = torch.log(torch.clamp(target[mask], min=self.eps))
        d = p - t
        
        return torch.sqrt((d**2).mean() - self.lam * (d.mean()**2)) * 10.0