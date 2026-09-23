import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_medical_vit_dir(dataset, model_dir=None):
    if model_dir:
        return model_dir
    if dataset in ("ISIC", "SD198"):
        env_name = "DERMLIP_DIR"
    elif dataset == "RFMiD":
        env_name = "RETCLIP_DIR"
    else:
        raise ValueError(
            "model=medical_vit is supported for ISIC, SD198, and RFMiD"
        )
    resolved = os.environ.get(env_name)
    if resolved:
        return resolved
    raise ValueError(
        "Set --medical_vit_dir or the {} environment variable".format(env_name)
    )


class PatchEmbed(nn.Module):
    def __init__(self, image_size=224, patch_size=16, width=768):
        super().__init__()
        self.proj = nn.Conv2d(
            3, width, kernel_size=patch_size, stride=patch_size, bias=False
        )

    def forward(self, values):
        return self.proj(values).flatten(2).transpose(1, 2)


class TimmMlp(nn.Module):
    def __init__(self, width=768):
        super().__init__()
        self.fc1 = nn.Linear(width, width * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(width * 4, width)

    def forward(self, values):
        return self.fc2(self.act(self.fc1(values)))


class TimmAttention(nn.Module):
    def __init__(self, width=768, heads=12):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)

    def forward(self, values):
        batch, length, width = values.shape
        qkv = self.qkv(values).reshape(
            batch, length, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        output = (attention.softmax(dim=-1) @ value).transpose(1, 2)
        return self.proj(output.reshape(batch, length, width))


class TimmBlock(nn.Module):
    def __init__(self, width=768, heads=12):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = TimmAttention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = TimmMlp(width)

    def forward(self, values):
        values = values + self.attn(self.norm1(values))
        return values + self.mlp(self.norm2(values))


class TimmLikeViT(nn.Module):
    def __init__(self, image_size=224, width=768, layers=12, heads=12):
        super().__init__()
        patch_count = (image_size // 16) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_count + 1, width))
        self.patch_embed = PatchEmbed(image_size=image_size, width=width)
        self.pos_drop = nn.Identity()
        self.blocks = nn.ModuleList(
            [TimmBlock(width, heads) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(width)


class ProjectionHead(nn.Module):
    def __init__(self, width=768, output_dim=512):
        super().__init__()
        self.proj = nn.Linear(width, output_dim)


class TimmOpenCLIPVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = TimmLikeViT()
        self.head = ProjectionHead()


class QuickGELU(nn.Module):
    def forward(self, values):
        return values * torch.sigmoid(1.702 * values)


class OpenCLIPResidualAttentionBlock(nn.Module):
    def __init__(self, width=768, heads=12, quick_gelu=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential()
        self.mlp.add_module("c_fc", nn.Linear(width, width * 4))
        activation = QuickGELU() if quick_gelu else nn.GELU()
        self.mlp.add_module("gelu", activation)
        self.mlp.add_module("c_proj", nn.Linear(width * 4, width))

    def forward(self, values):
        normalized = self.ln_1(values)
        values = values + self.attn(
            normalized, normalized, normalized, need_weights=False
        )[0]
        return values + self.mlp(self.ln_2(values))


class OpenCLIPTransformer(nn.Module):
    def __init__(self, width=768, layers=12, heads=12, quick_gelu=False):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [
                OpenCLIPResidualAttentionBlock(width, heads, quick_gelu)
                for _ in range(layers)
            ]
        )

    def forward(self, values):
        for block in self.resblocks:
            values = block(values)
        return values


class OpenCLIPVisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        quick_gelu=False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, width, kernel_size=patch_size, stride=patch_size, bias=False
        )
        scale = width ** -0.5
        grid_size = image_size // patch_size
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(grid_size ** 2 + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = OpenCLIPTransformer(
            width, layers, heads, quick_gelu=quick_gelu
        )
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))


def _checkpoint_state(model_dir):
    if not model_dir:
        raise ValueError("--medical_vit_dir is required for model=medical_vit")
    if os.path.isfile(model_dir):
        checkpoint = model_dir
    else:
        candidates = (
            os.path.join(model_dir, "ret-clip.pt"),
            os.path.join(model_dir, "open_clip_pytorch_model.bin"),
            os.path.join(model_dir, "open_clip_model.safetensors"),
        )
        checkpoint = next(
            (path for path in candidates if os.path.isfile(path)), None
        )
        if checkpoint is None:
            raise FileNotFoundError(
                "no supported medical ViT checkpoint in {}".format(model_dir)
            )
    if checkpoint.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                "safetensors is required to load {}; use the .bin checkpoint "
                "or install safetensors".format(checkpoint)
            )
        state = load_file(checkpoint, device="cpu")
    else:
        state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError("medical ViT checkpoint must contain a state dictionary")
    return state, checkpoint


def _visual_state(state):
    output = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if not key.startswith("visual."):
            continue
        key = key[len("visual."):]
        if ".attn.Wqkv.weight" in key:
            key = key.replace(".attn.Wqkv.weight", ".attn.in_proj_weight")
        elif ".attn.Wqkv.bias" in key:
            key = key.replace(".attn.Wqkv.bias", ".attn.in_proj_bias")
        if key.startswith("head.") and not key.startswith("head.proj."):
            key = "head.proj." + key[len("head."):]
        if key.startswith("trunk.head."):
            key = "head.proj." + key[len("trunk.head."):]
        output[key] = value
    return output


class MedicalVisionBackbone(nn.Module):
    output_dim = 512

    def __init__(self, model_dir):
        super().__init__()
        state, checkpoint = _checkpoint_state(model_dir)
        visual_state = _visual_state(state)
        if "conv1.weight" in visual_state:
            quick_gelu = os.path.basename(checkpoint) == "ret-clip.pt"
            self.visual = OpenCLIPVisionTransformer(quick_gelu=quick_gelu)
            self.kind = "openclip_vit"
            required = (
                "conv1.weight",
                "transformer.resblocks.0.attn.in_proj_weight",
                "proj",
            )
        elif "trunk.patch_embed.proj.weight" in visual_state:
            self.visual = TimmOpenCLIPVisual()
            self.kind = "timm_vit"
            required = (
                "trunk.patch_embed.proj.weight",
                "trunk.blocks.0.attn.qkv.weight",
                "head.proj.weight",
            )
        else:
            raise RuntimeError(
                "unsupported medical visual checkpoint keys: {}".format(
                    sorted(visual_state)[:8]
                )
            )
        missing, _ = self.visual.load_state_dict(visual_state, strict=False)
        missing_required = [key for key in required if key in missing]
        if missing_required:
            raise RuntimeError(
                "missing required medical ViT weights: {}".format(
                    missing_required
                )
            )
        self.checkpoint = checkpoint

    def _openclip_forward(self, images):
        model = self.visual
        values = model.conv1(images)
        values = values.reshape(
            values.shape[0], values.shape[1], -1
        ).permute(0, 2, 1)
        cls = model.class_embedding.to(values.dtype).reshape(1, 1, -1)
        values = torch.cat(
            (cls.expand(values.shape[0], -1, -1), values), dim=1
        )
        values = model.ln_pre(
            values + model.positional_embedding.to(values.dtype)
        )
        values = model.transformer(values.permute(1, 0, 2))
        values = model.ln_post(values.permute(1, 0, 2))
        return values[:, 0] @ model.proj

    def _timm_forward(self, images):
        trunk = self.visual.trunk
        values = trunk.patch_embed(images)
        cls = trunk.cls_token.expand(values.shape[0], -1, -1)
        values = trunk.pos_drop(
            torch.cat((cls, values), dim=1) + trunk.pos_embed
        )
        for block in trunk.blocks:
            values = block(values)
        values = trunk.norm(values)
        return self.visual.head.proj(values[:, 0])

    def forward(self, images):
        if images.shape[-2:] != (224, 224):
            raise ValueError(
                "medical_vit requires 224x224 inputs, got {}".format(
                    tuple(images.shape[-2:])
                )
            )
        if self.kind == "openclip_vit":
            return self._openclip_forward(images)
        return self._timm_forward(images)


class ResidualAdapter(nn.Module):
    def __init__(self, feature_dim=512, bottleneck_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, feature_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, values):
        return values + self.up(self.act(self.down(self.norm(values))))


class MedicalViTEncoder(nn.Module):
    output_dim = 512

    def __init__(
        self,
        model_dir,
        train_mode="full",
        adapter_dim=128,
        shared_backbone=None,
        normalize_output=True,
    ):
        super().__init__()
        if train_mode not in ("adapter", "full"):
            raise ValueError("vit_train_mode must be adapter or full")
        self.train_mode = train_mode
        self.normalize_output = normalize_output
        self.backbone = (
            shared_backbone
            if shared_backbone is not None
            else MedicalVisionBackbone(model_dir)
        )
        if train_mode == "adapter":
            self.adapter = ResidualAdapter(
                feature_dim=self.output_dim, bottleneck_dim=adapter_dim
            )
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()
        else:
            self.adapter = nn.Identity()
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True

    @property
    def checkpoint(self):
        return self.backbone.checkpoint

    def train(self, mode=True):
        super().train(mode)
        if self.train_mode == "adapter":
            self.backbone.eval()
        return self

    def forward(self, images):
        if self.train_mode == "adapter":
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)
        features = self.adapter(features)
        if self.normalize_output:
            features = F.normalize(features, dim=-1)
        return features
