import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure inputs are probabilities
        if not (inputs.min() >= 0 and inputs.max() <= 1):
            raise ValueError("Inputs should be probabilities (values between 0 and 1).")

        # Compute binary cross entropy loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)

        # Compute focal loss
        focal_loss = (1 - pt) ** self.gamma * bce_loss

        # Apply alpha weighting
        if self.alpha >= 0:
            alpha_t = self.alpha * (1 - targets) + (1 - self.alpha) * targets
            focal_loss = alpha_t * focal_loss

        # Reduce loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



# 构造损失
def build_criterion(criterion, gamma=2, alpha=0.15,c_margin=1.4):
    if criterion == "FocalLoss":
        print("Loss : Focal loss")
        return FocalLoss(gamma=gamma, alpha=0.15, reduction='mean')
    else:
        raise NotImplementedError