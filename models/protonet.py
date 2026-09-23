from hyptorch.nn import ToPoincare
from hyptorch.pmath import zero_dist_matrix

import torch
import torch.nn as nn
from networks.ResNet import resnet50
from networks.ResNet12 import Res12
from networks.DenseNet import densenet121
from networks.MedicalViT import MedicalViTEncoder
from torch.autograd import Variable

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class ProtoNet(nn.Module):
    def __init__(self, args, shared_vit_backbone=None):
        super().__init__()
        self.args = args
        model_name = args.model

        if model_name == "resnet50":
            self.encoder = resnet50(remove_linear=True)
        elif model_name == "densenet121":
            self.encoder = densenet121(remove_linear=True)
        elif model_name == "resnet12":
            self.encoder = Res12()
        elif model_name == "medical_vit":
            self.encoder = MedicalViTEncoder(
                model_dir=args.medical_vit_dir,
                train_mode=args.vit_train_mode,
                adapter_dim=args.vit_adapter_dim,
                shared_backbone=shared_vit_backbone,
                normalize_output=not args.hyperbolic,
            )
        else:
            raise ValueError("Model not found")

        if args.hyperbolic:
            self.e2p = ToPoincare(
                c=args.c,
                train_c=args.train_c,
                train_x=args.train_x,
                riemannian=args.riemannian,
            )

    def forward(self, data_shot, data_query):
        if self.args.model == "medical_vit":
            shot_count = data_shot.shape[0]
            features = self.encoder(torch.cat((data_shot, data_query), dim=0))
            proto = features[:shot_count]
            data_query = features[shot_count:]
        else:
            proto = self.encoder(data_shot)
            data_query = self.encoder(data_query)
        shot_zero = None
        if self.args.hyperbolic:
            proto = 5e-2 * proto
            proto = self.e2p(proto)
            data_query = 5e-2 * data_query
            data_query = self.e2p(data_query)
            shot_zero = (
                    zero_dist_matrix(proto, c=self.e2p.c) / self.args.temperature
            )

        return proto, data_query, shot_zero

    def params(self):
        for name, param in self.named_params(self):
            if param.requires_grad:
                yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        source_params = nn.ParameterList(source_params)
        if source_params is not None:
            trainable_params = (
                item for item in self.named_params(self) if item[1].requires_grad
            )
            for tgt, src in zip(trainable_params, source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not param.requires_grad:
                    continue
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if isinstance(param, torch.Tensor):
            param = torch.nn.Parameter(param)

        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())
