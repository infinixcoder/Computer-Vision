import torch
import torch.nn as nn
import torchvision.models as models
import timm

# --- Subtask 1.2: SE Block & SE-ResNet ---
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResNet18_SE(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_SE, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.conv1, self.bn1, self.relu, self.maxpool = resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.se1 = SEBlock(64)
        self.layer2 = resnet.layer2
        self.se2 = SEBlock(128)
        self.layer3 = resnet.layer3
        self.se3 = SEBlock(256)
        self.layer4 = resnet.layer4
        self.se4 = SEBlock(512)
        
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.se1(self.layer1(x))
        x = self.se2(self.layer2(x))
        x = self.se3(self.layer3(x))
        x = self.se4(self.layer4(x))
        return self.fc(torch.flatten(self.avgpool(x), 1))

# --- Subtask 2.2: Dynamic Tanh ---
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape):
        super(DynamicTanh, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.alpha = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        return torch.tanh(self.alpha * x)

def replace_layernorm_with_dyt(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, DynamicTanh(child.normalized_shape))
        else:
            replace_layernorm_with_dyt(child)

# --- Main Model Factory ---
def get_model(model_name, num_classes=10):
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet18_se":
        model = ResNet18_SE(num_classes=num_classes)
    elif model_name == "deit3":
        model = timm.create_model('deit3_small_patch16_224', pretrained=True, num_classes=num_classes)
    elif model_name == "deit3_dyt":
        model = timm.create_model('deit3_small_patch16_224', pretrained=True, num_classes=num_classes)
        replace_layernorm_with_dyt(model)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model