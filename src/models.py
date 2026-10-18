
import torch.nn as nn

def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )

class BaselineCNN(nn.Module):
    """Small CNN from scratch: 4 conv blocks -> global average pool -> 1 logit."""
    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 16), conv_block(16, 32), conv_block(32, 64), conv_block(64, 128)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.pool(self.features(x))).squeeze(1)   # raw logit, shape (B,)


class DenseNetClassifier(nn.Module):
    """ImageNet-pretrained DenseNet121 with a new 1-logit head."""
    def __init__(self, pretrained=True, dropout=0.2):
        super().__init__()
        from torchvision import models as tvm
        weights = tvm.DenseNet121_Weights.DEFAULT if pretrained else None
        self.net = tvm.densenet121(weights=weights)
        self.net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.net.classifier.in_features, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)
