from torch import Tensor, nn
import torch

class VPULoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        temp = (self.alpha * logits.std(0, False)).clamp(0.1, 1.0).detach()

        preds = (logits / temp).sigmoid()
        log_pos_probs = preds.clamp_min(1e-5).log() * labels

        num_p_rec = 1 / labels.sum(0)
        num_p_rec = torch.where(num_p_rec.isinf(), torch.zeros_like(num_p_rec), num_p_rec)
        num_u_rec = 1 / labels.shape[0]

        p_c = preds.mean(0)
        u_loss = (num_u_rec * preds.sum(0)).clamp_min(1e-5).log()
        p_loss = num_p_rec * log_pos_probs.sum(0)

        class_losses = p_c.pow(self.gamma).detach() * u_loss - p_loss

        return class_losses


class MixupLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.beta_dist = torch.distributions.Beta(alpha, alpha)

    def forward(self, x: Tensor, preds: Tensor, y: Tensor, model):
        midpoint1 = int(x.shape[0] / 2)
        # This makes sure all partitions are the same size
        midpoint2 = midpoint1 + (1 if x.shape[0] % 2 == 1 else 0)

        x1, x2 = x[:midpoint1], x[midpoint2:] # logits
        y1, y2 = y[:midpoint1], y[midpoint2:] # labels
        yh1, yh2 = preds[:midpoint1], preds[midpoint2:] # preds

        target1 = y1 + yh1 * (y1 == 0)
        target2 = y2 + yh2 * (y2 == 0)

        idx_perm = torch.randperm(len(x1))

        lam = self.beta_dist.sample()

        x_mixed = x1[idx_perm]      * lam + x2      * (1 - lam)
        y_mixed = target1[idx_perm] * lam + target2 * (1 - lam)

        with torch.autocast('cuda', torch.bfloat16):
            outputs = model(x_mixed).float()

        log_outputs = outputs.sigmoid().clamp_min(1e-10).log().float()

        return (((y_mixed.clamp_min(1e-10).log() - log_outputs).pow(2).sum(dim=0)) * (1 / len(x1))).sum()
