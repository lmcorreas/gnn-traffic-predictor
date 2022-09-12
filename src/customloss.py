import torch
import torch.nn as nn

from torch import Tensor
    
class TrafficLoss(nn.Module):
    def __init__(self, out_limits_factor = 1000.0):
        super(TrafficLoss, self).__init__()
        self.out_limits_factor = out_limits_factor
        
    def forward(self, input: Tensor, target: Tensor, mask: Tensor, factor: float) -> Tensor:
        ret = 0.0
        
        if factor > 0:
            # Valores activos dentro de la máscara de existencia de datos reales
            masked = torch.mul(torch.mean((input[mask]-target[mask])**2), factor)
            
            numel = (~mask).sum().item()
            
            # Sin información real, se aplica control de rango
            below_zero = (input < 0)
            below = torch.mul(torch.div(torch.sum((input[((~mask) & below_zero)] ** 2)), numel), self.out_limits_factor * factor)
            
            over_max = (input > 100)
            over = torch.mul(torch.div(torch.sum(((input[((~mask) & over_max)] - 1) ** 2)), numel), self.out_limits_factor * factor)

            ret = masked + below + over
        
        return ret