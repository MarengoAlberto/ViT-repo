import torch
from torchvision.io import read_image
import torchvision.transforms as T
from ts.torch_handler.base_handler import BaseHandler

from predictor.model import VisionTransformer
from src.lightning_model import ViT

model_kwargs={
            "embed_dim": 256,
            "hidden_dim": 512,
            "num_heads": 8,
            "num_layers": 6,
            "patch_size": 4,
            "num_channels": 3,
            "num_patches": 64,
            "num_classes": 10,
            "dropout": 0.2,
        }
model_pt_path = 'saved_models/VisionTransformers/ViT/epoch=1-step=702.ckpt'
light_model = ViT(model_kwargs, 0.001)
# model = VisionTransformer(**model_kwargs)
light_model.load_from_checkpoint(checkpoint_path=model_pt_path)
# model = torch.load(model_pt_path)
model = light_model.model
torch.save(model.state_dict(), "ViT.pt")
new_model = VisionTransformer(**model_kwargs)
new_model.load_state_dict(torch.load("ViT.pt"))
new_model.eval()
print('Success')