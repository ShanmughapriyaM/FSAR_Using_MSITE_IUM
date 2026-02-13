import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone="resnet18", device="cpu"):
        super().__init__()

        if backbone == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.feat_dim = 512
        elif backbone == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.feat_dim = 2048
        else:
            raise ValueError("Unsupported backbone")

        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.backbone.to(device).eval()

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.device = device

    @torch.no_grad()
    def forward(self, x):
        x = self.backbone(x)
        return x.view(x.size(0), -1).squeeze(0)


class ResNetTemporalDifference:
    def __init__(self, backbone="resnet18", device="cpu"):
        self.device = device
        self.extractor = ResNetFeatureExtractor(backbone, device)
        self.prev_feat = None
        self.out_dim = self.extractor.feat_dim * 2

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def reset(self):
        self.prev_feat = None

    @torch.no_grad()
    def extract(self, frame):
        x = self.transform(frame).unsqueeze(0).to(self.device)
        f_t = self.extractor(x)

        if self.prev_feat is None:
            delta = torch.zeros_like(f_t)
        else:
            delta = f_t - self.prev_feat

        self.prev_feat = f_t
        return torch.cat([f_t, delta], dim=0)
