import argparse
import os.path as osp

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataloader.samplers import CategoriesSampler
from hyptorch.pmath import dist_matrix
from models.protonet import ProtoNet
from networks.MedicalViT import resolve_medical_vit_dir
from utils import (
    compute_confidence_interval,
    count_acc,
    set_gpu,
    set_random_seed,
)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="resnet12",
        choices=["resnet12", "resnet50", "densenet121", "medical_vit"]
    )
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=5)
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--validation_way", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--mlp_lr", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument(
        "--dataset", type=str, default="ISIC", choices=["ISIC", "SD198", "RFMiD"]
    )
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="epoch-last.pth")
    parser.add_argument("--hyperbolic", action="store_true", default=False)
    parser.add_argument("--c", type=float, default=None)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--init_weights", type=str, default=None)
    parser.add_argument("--gpu", default="0,1,2,3")
    parser.add_argument("--train_c", action="store_true", default=False)
    parser.add_argument("--train_x", action="store_true", default=False)
    parser.add_argument("--not-riemannian", action="store_true")
    parser.add_argument("--sampler_batch", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test_episodes", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--isic_image_path", type=str, default=None)
    parser.add_argument("--isic_split_path", type=str, default="./data/isic/split")
    parser.add_argument("--sd198_image_path", type=str, default=None)
    parser.add_argument("--sd198_split_path", type=str, default=None)
    parser.add_argument("--rfmid_image_path", type=str, default=None)
    parser.add_argument("--rfmid_split_path", type=str, default="./data/RFMiD/split")
    parser.add_argument("--medical_vit_dir", type=str, default=None)
    parser.add_argument(
        "--vit_train_mode",
        type=str,
        default="full",
        choices=["adapter", "full"],
    )
    parser.add_argument("--vit_adapter_dim", type=int, default=128)
    return parser


def get_dataset(args):
    if args.dataset == "ISIC":
        from dataloader.isic import ISIC as Dataset
    elif args.dataset == "SD198":
        from dataloader.sd198 import SD198 as Dataset
    elif args.dataset == "RFMiD":
        from dataloader.rfmid import RFMiD as Dataset
    else:
        raise ValueError("Non-supported Dataset.")
    return Dataset


def prototype_logits(args, feature_query, train_prototypes):
    if args.hyperbolic:
        return -dist_matrix(feature_query, train_prototypes, c=args.c) / args.temperature
    return -1 * torch.cdist(feature_query, train_prototypes)


def episode_auc(label_query, prob):
    label_np = label_query.detach().cpu().numpy()
    prob_np = prob.detach().cpu().numpy()
    try:
        if prob_np.shape[1] == 2:
            return roc_auc_score(label_np, prob_np[:, 1])
        return roc_auc_score(label_np, prob_np, average="macro", multi_class="ovo")
    except ValueError:
        return np.nan


def evaluate(args):
    if args.c is None:
        args.c = 1e-3 if args.shot == 1 else 5e-3
    set_gpu(args)
    args.riemannian = not args.not_riemannian
    if args.model == "medical_vit":
        args.medical_vit_dir = resolve_medical_vit_dir(
            args.dataset, args.medical_vit_dir
        )
    if args.seed is not None:
        set_random_seed(args.seed)

    Dataset = get_dataset(args)
    model = ProtoNet(args)
    checkpoint = torch.load(
        osp.join(args.save_path, args.checkpoint), map_location="cpu"
    )
    if checkpoint.get("checkpoint_format") == "trainable_only":
        expected = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        actual = set(checkpoint["params"])
        if actual != expected:
            raise RuntimeError(
                "trainable-only checkpoint keys do not match the model: "
                "missing={}, unexpected={}".format(
                    sorted(expected - actual), sorted(actual - expected)
                )
            )
        model.load_state_dict(checkpoint["params"], strict=False)
    else:
        model.load_state_dict(checkpoint["params"])
    model = model.cuda()
    model.eval()

    test_set = Dataset("test", args)
    test_way = args.validation_way
    sampler = CategoriesSampler(
        test_set.label, args.test_episodes, test_way, args.shot + args.query
    )
    loader = DataLoader(
        test_set, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True
    )
    acc_record = np.zeros((args.test_episodes,))
    auc_record = []

    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * test_way
            data_shot, data_query = data[:p], data[p:]
            label_shot = torch.arange(test_way).repeat(args.shot).type(torch.cuda.LongTensor)
            label_query = torch.arange(test_way).repeat(args.query).type(torch.cuda.LongTensor)

            feature_shot, feature_query, _ = model(data_shot, data_query)
            centroid_classes = torch.unique(label_shot)
            train_prototypes = torch.stack(
                [feature_shot[torch.where(label_shot == c)[0]].mean(0) for c in centroid_classes]
            )
            logits = prototype_logits(args, feature_query, train_prototypes)
            acc = count_acc(logits, label_query)
            prob = torch.softmax(logits, dim=1)

            acc_record[i - 1] = acc
            auc_record.append(episode_auc(label_query, prob))

    acc_mean, acc_ci = compute_confidence_interval(acc_record)
    auc_record = np.array(auc_record, dtype=float)
    auc_record = auc_record[~np.isnan(auc_record)]
    if len(auc_record) == 0:
        auc_mean, auc_ci = np.nan, np.nan
    else:
        auc_mean, auc_ci = compute_confidence_interval(auc_record)
    return {
        "seed": args.seed,
        "acc": acc_mean * 100.0,
        "acc_ci": acc_ci * 100.0,
        "auc": auc_mean * 100.0,
        "auc_ci": auc_ci * 100.0,
    }


if __name__ == "__main__":
    result = evaluate(build_parser().parse_args())
    print("seed,acc,acc_ci,auc,auc_ci")
    print("{seed},{acc:.4f},{acc_ci:.4f},{auc:.4f},{auc_ci:.4f}".format(**result))
