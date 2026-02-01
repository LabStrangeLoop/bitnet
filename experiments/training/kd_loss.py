"""Knowledge Distillation loss function."""

import torch
from torch import nn
from torch.nn import functional


class KDLoss(nn.Module):
    """Combined cross-entropy and distillation loss.

    Standard KD loss: α * soft_loss + (1-α) * hard_loss
    where soft_loss = T² * KL(student_soft, teacher_soft)
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.9):
        """
        Args:
            temperature: Softmax temperature for soft targets (typically 4-20).
            alpha: Weight for distillation loss (1-alpha for hard targets).
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute KD loss.

        Returns:
            Tuple of (total_loss, metrics_dict with individual losses).
        """
        # Hard target loss (standard cross-entropy)
        hard_loss = functional.cross_entropy(student_logits, targets)

        # Soft target loss (KL divergence with temperature scaling)
        soft_student = functional.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = functional.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = functional.kl_div(soft_student, soft_teacher, reduction="batchmean")

        # Scale by T² (standard practice to match gradient magnitudes)
        soft_loss = soft_loss * (self.temperature**2)

        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        metrics = {
            "hard_loss": hard_loss.item(),
            "soft_loss": soft_loss.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, metrics
