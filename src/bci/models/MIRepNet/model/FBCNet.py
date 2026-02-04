import torch
import torch.nn as nn
from torch.nn import functional as F


class FBCNet(nn.Module):
    """
    Filter Bank Common Spatial Pattern Network (FBCNet).
    
    Standard implementation that outputs logits (use CrossEntropyLoss).
    Expects input shape: (batch, n_bands, n_electrodes, time_points)
    """
    def __init__(self,
                 n_classes,
                 input_shape,
                 m,
                 temporal_stride,
                 weight_init_method=None,
                 ):
        super().__init__()
        self.temporal_stride = temporal_stride

        batch_size, n_band, n_electrode, time_points = input_shape

        # SCB (Spatial Convolution Block)
        self.scb = nn.Sequential(
            Conv2dWithConstraint(n_band, m * n_band, (n_electrode, 1), groups=n_band, max_norm=2),
            nn.BatchNorm2d(m * n_band),
            swish()
        )

        # Temporal Layer
        self.temporal_layer = LogVarLayer(-1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(n_band * m * temporal_stride, n_classes, max_norm=0.5)
        )

        initialize_weight(self, weight_init_method)

    def forward(self, x):
        out = self.scb(x)
        out = out.reshape([*out.shape[:2], self.temporal_stride, int(out.shape[-1] / self.temporal_stride)])
        out = self.temporal_layer(out)
        out = self.classifier(out)
        return out
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6))

def initialize_weight(model, method):
    method = dict(normal=['normal_', dict(mean=0, std=0.01)],
                  xavier_uni=['xavier_uniform_', dict()],
                  xavier_normal=['xavier_normal_', dict()],
                  he_uni=['kaiming_uniform_', dict()],
                  he_normal=['kaiming_normal_', dict()]).get(method)
    if method is None:
        return None

    for module in model.modules():
        # LSTM
        if module.__class__.__name__ in ['LSTM']:
            for param in module._all_weights[0]:
                if param.startswith('weight'):
                    getattr(nn.init, method[0])(getattr(module, param), **method[1])
                elif param.startswith('bias'):
                    nn.init.constant_(getattr(module, param), 0)
        else:
            if hasattr(module, "weight"):
                # Not BN
                if not ("BatchNorm" in module.__class__.__name__):
                    getattr(nn.init, method[0])(module.weight, **method[1])
                # BN
                else:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, "bias"):
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

