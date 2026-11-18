"""Single-output image score regressor, with a frozen ImageNet ResNet-18 encoder."""
from pathlib import Path
import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.transforms import functional as TF
from PIL import Image, ImageOps

class ScoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.encoder.requires_grad_(False)
        self.register_buffer('feature_mean', torch.zeros(512))
        self.register_buffer('feature_scale', torch.ones(512))
        self.head = nn.Sequential(nn.Linear(512,64), nn.ReLU(), nn.Linear(64,1), nn.Sigmoid())

    def encode(self, images):
        return self.encoder(images)

    def score_features(self, features):
        normalized = (features-self.feature_mean)/self.feature_scale
        return self.head(normalized).squeeze(-1)*100

    def forward(self, images):
        return self.score_features(self.encode(images))

class DimensionScoreModel(ScoreModel):
    """Four rubric outputs (two abdomen rubrics) and three assessability logits."""
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(512,64),nn.ReLU())
        self.regression = nn.Linear(64,4)
        self.assessability = nn.Linear(64,3)

    def score_features(self, features):
        hidden=self.head((features-self.feature_mean)/self.feature_scale)
        return {'scores':torch.sigmoid(self.regression(hidden))*100,
                'assessability_logits':self.assessability(hidden)}

def masked_score_loss(predictions, targets, mask):
    """Average the per-rubric MSE over rubrics with labels; NA has zero gradient."""
    mask=mask.bool()
    squared=torch.where(mask,((predictions-targets)/100).square(),torch.zeros_like(predictions))
    counts=mask.sum(0)
    per_rubric=squared.sum(0)/counts.clamp_min(1)
    return per_rubric.sum()/(counts>0).sum().clamp_min(1)

def format_dimension_result(scores, logits, mode='abs', threshold=0.5, trained=None):
    if mode not in ('abs','lines') or not 0<=threshold<=1:
        raise ValueError('Invalid rubric or assessability threshold')
    if not torch.isfinite(scores).all() or not torch.isfinite(logits).all():
        raise ValueError('Non-finite model output')
    indices=[0 if mode=='abs' else 1,2,3]
    flags=(torch.sigmoid(logits)>=threshold).tolist()
    trained=trained if trained is not None else [True]*4
    values=[round(float(scores[index]),1) if flags[i] and trained[index] else 'NA'
            for i,index in enumerate(indices)]
    valid=[v for v in values if isinstance(v,(int,float))]
    return {'abdomen_score':values[0],'shoulder_score':values[1],'arm_score':values[2],
            'total_score':round(sum(valid)/len(valid),1) if valid else 'NA',
            'abdomen_mode':mode}

def load_image(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        # Preserve body proportions and the complete field of view.
        image = ImageOps.pad(image,(224,224),method=Image.Resampling.BILINEAR,color=(128,128,128))
        tensor = TF.to_tensor(image)
    return TF.normalize(tensor,[0.485,0.456,0.406],[0.229,0.224,0.225])

def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(Path(path),map_location=device,weights_only=True)
    if checkpoint.get('format_version') not in (1,2):
        raise ValueError('Unsupported checkpoint version')
    model = (DimensionScoreModel() if checkpoint['format_version']==2 else ScoreModel()).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, checkpoint
