import numpy as np
import torch
from PIL import ImageOps
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.transforms.functional import pil_to_tensor

CNN = {'resnet18': ('ResNet18_Weights', 'IMAGENET1K_V1', 512),
       'resnet50': ('ResNet50_Weights', 'IMAGENET1K_V2', 2048),
       'mobilenet_v3_small': ('MobileNet_V3_Small_Weights', 'IMAGENET1K_V1', 576),
       'convnext_tiny': ('ConvNeXt_Tiny_Weights', 'IMAGENET1K_V1', 768)}
HF = {'dinov2': 'facebook/dinov2-base',
      'dinov3': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
      'siglip2': 'google/siglip2-base-patch16-naflex',
      'mae': 'facebook/vit-mae-base'}
ENCODERS = list(CNN) + list(HF)


class Encoder(nn.Module):
    def __init__(self, name, size=(224, 448), grid=(2, 4), revision='main'):
        super().__init__()
        self.name, self.size, self.grid = name, tuple(size), tuple(grid)
        self.mean, self.std = [.485, .456, .406], [.229, .224, .225]
        if name in CNN:
            enum, version, self.dim = CNN[name]
            weights = getattr(getattr(models, enum), version)
            model = getattr(models, name)(weights=weights)
            self.model = nn.Sequential(*list(model.children())[:-2]) if name.startswith('resnet') else model.features
            self.norm = model.classifier[0] if name == 'convnext_tiny' else nn.Identity()
            self.revision = f'torchvision:{enum}.{version}'
        else:
            from transformers import AutoImageProcessor, AutoModel, Siglip2VisionModel, ViTMAEModel
            self.processor = AutoImageProcessor.from_pretrained(HF[name], revision=revision)
            cls = {'siglip2': Siglip2VisionModel, 'mae': ViTMAEModel}.get(name, AutoModel)
            self.model = cls.from_pretrained(HF[name], revision=revision)
            self.dim = self.model.config.hidden_size
            self.patch = self.model.config.patch_size
            self.revision = self.model.config._commit_hash
            self.mean, self.std = self.processor.image_mean, self.processor.image_std
            assert all(s % self.patch == 0 for s in self.size)
            if name == 'mae':
                self.model.config.mask_ratio = 0.

    def config(self):
        return {'name': self.name, 'size': self.size, 'grid': self.grid,
                'revision': self.revision, 'checkpoint': HF.get(self.name, self.name)}

    def preprocess(self, images):
        fill = tuple(round(v*255) for v in self.mean)
        images = [ImageOps.pad(im, self.size[::-1], color=fill) for im in images]
        if self.name == 'siglip2':
            return dict(self.processor(images=images, max_num_patches=np.prod(self.size)//self.patch**2,
                                       return_tensors='pt'))
        x = torch.stack([pil_to_tensor(im).float()/255 for im in images])
        mean = torch.tensor(self.mean)[None, :, None, None]
        std = torch.tensor(self.std)[None, :, None, None]
        return {'pixel_values': (x-mean)/std}

    def forward(self, inputs, pool=None):
        if self.name in CNN:
            maps = self.model(inputs['pixel_values'])
            global_feature = self.norm(maps.mean((-2, -1), keepdim=True)).flatten(1)
            maps = self.norm(maps)
            means = maps.mean((-2, -1))
        else:
            if self.name == 'mae':
                count = np.prod(self.size)//self.patch**2
                noise = torch.arange(count, device=inputs['pixel_values'].device).expand(len(inputs['pixel_values']), -1)
                out = self.model(**inputs, noise=noise, interpolate_pos_encoding=True)
            else:
                out = self.model(**inputs)
            if self.name == 'siglip2':
                maps = torch.stack([F.adaptive_avg_pool2d(tokens[:h*w].T.reshape(self.dim, h, w), self.grid)
                                    for tokens, (h, w) in zip(out.last_hidden_state, inputs['spatial_shapes'].tolist())])
                mask = inputs['pixel_attention_mask'].unsqueeze(-1)
                means = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
                global_feature = out.pooler_output
            else:
                offset = 1 + getattr(self.model.config, 'num_register_tokens', 0)
                tokens = out.last_hidden_state[:, offset:]
                maps = tokens.transpose(1, 2).reshape(-1, self.dim, self.size[0]//self.patch, self.size[1]//self.patch)
                global_feature, means = out.last_hidden_state[:, 0], tokens.mean(1)
        features = {'global': global_feature, 'mean': means,
                    'spatial': F.adaptive_avg_pool2d(maps, self.grid).flatten(1)}
        return features if pool is None else features[pool]

    def tune(self, mode):
        self.requires_grad_(mode == 'full')
        if mode == 'partial':
            if self.name in CNN:
                blocks = list(self.model.children())[-2:] + [self.norm]
            elif self.name == 'dinov3':
                blocks = list(self.model.layer[-2:]) + [self.model.norm]
            elif self.name == 'siglip2':
                m = self.model.vision_model
                blocks = list(m.encoder.layers[-2:]) + [m.post_layernorm, m.head]
            else:
                blocks = list(self.model.encoder.layer[-2:]) + [self.model.layernorm]
            for block in blocks:
                block.requires_grad_(True)


def head(dimension, outputs):
    return nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 128), nn.GELU(),
                         nn.Linear(128, outputs), nn.Sigmoid())
