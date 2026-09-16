from transformers import AutoModel
from torch import nn
import torch, dotenv, os

dotenv.load_dotenv()
token = os.getenv('HF_TOKEN')


class Tagger(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_freeze = True

        self.backbone: nn.Module = AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m', token=token)
        self.backbone.requires_grad_(False)
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 35)
        )

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

        last_layer = self.classifier[-1]
        nn.init.xavier_uniform_(last_layer.weight, gain=0.1)
        nn.init.zeros_(last_layer.bias)

    def forward(self, x):
        if (self.backbone_freeze):
            self.backbone.eval()

            with torch.no_grad():
                embeddings = self.backbone(x).last_hidden_state[:, 0]
        else:
            embeddings = self.backbone(x).last_hidden_state[:, 0]

        return self.classifier(embeddings)

    def freeze_backbone(self, mode=True):
        self.backbone_freeze = mode

        for layer in self.backbone.model.layer[-4:]:
            layer.requires_grad_(not mode)
