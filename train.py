import argparse
import os.path as osp
import torch
import torch.nn.functional as F
from torch.func import functional_call
from dataloader.samplers import CategoriesSampler
from models.protonet import ProtoNet
try:
    from tensorboardX import SummaryWriter
except ImportError:
    class SummaryWriter(object):
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass
from torch.utils.data import DataLoader
from networks.MlP import MlP
from networks.MedicalViT import resolve_medical_vit_dir
import csv
from utils import (
    pprint,
    set_gpu,
    set_random_seed,
    ensure_path,
    Averager,
    Timer,
    count_acc,
)
from hyptorch.pmath import dist_matrix

if __name__ == "__main__":

    def logits_label(feature_query, train_prototypes, label_query, centroid_classes, epsilons, way):
        if args.hyperbolic:
            logits_ori = (
                    -dist_matrix(feature_query, train_prototypes, c=args.c) / args.temperature)
        else:
            logits_ori = -1 * torch.cdist(feature_query, train_prototypes)
        mask = torch.eq(label_query.contiguous().view(-1, 1), centroid_classes.contiguous().view(-1, 1).T).cuda()
        mask_false = ~mask
        # Build a complementary one-hot mask for all non-target classes.
        ones = torch.sparse.torch.eye(way).cuda()
        lable_onehot = ones.index_select(0, label_query)
        label_false = ~(lable_onehot.bool())
        a = logits_ori.clone()
        logits_label1 = logits_ori.clone()
        logits_label2 = logits_ori.clone()
        logits_ori[mask] = logits_ori[mask] - epsilons[label_query].cuda()
        epsilons_new = epsilons.contiguous().view(1, -1).repeat(label_query.shape[0], 1)
        logits_ori[mask_false] = a[mask_false] + epsilons_new[label_false].cuda()

        logits_label2 = logits_label2 - epsilons.contiguous().view(1, -1).repeat(label_query.shape[0], 1).cuda()

        return logits_label1, logits_label2, logits_ori,

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument(
        "--model",
        type=str,
        default="resnet12",
        choices=["resnet12", "resnet50", "densenet121", "medical_vit"],
    )
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=5)
    parser.add_argument("--way", type=int, default=3)
    parser.add_argument("--validation_way", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--mlp_lr", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument(
        "--dataset", type=str, default="ISIC", choices=["ISIC", "SD198", "RFMiD"]
    )
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--hyperbolic", action="store_true", default=False)
    parser.add_argument("--c", type=float, default=None)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--init_weights", type=str, default=None)
    parser.add_argument('--gpu', default='0,1,2,3')
    parser.add_argument("--train_c", action="store_true", default=False)
    parser.add_argument("--train_x", action="store_true", default=False)
    parser.add_argument("--not-riemannian", action="store_true")
    parser.add_argument("--sampler_batch", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epsilons_ratio", type=float, default=0.0)
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
    args = parser.parse_args()
    if args.c is None:
        args.c = 1e-3 if args.shot == 1 else 5e-3
    if args.model == "medical_vit":
        args.medical_vit_dir = resolve_medical_vit_dir(
            args.dataset, args.medical_vit_dir
        )
    set_gpu(args)
    pprint(vars(args))
    args.riemannian = not args.not_riemannian

    if torch.cuda.is_available():
        print("CUDA IS AVAILABLE")

    medical_sdp_state = None
    if args.model == "medical_vit" and torch.cuda.is_available():
        medical_sdp_state = (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        )
        print("medical ViT meta attention: math SDP for second-order gradients")

    if args.seed != None:
        set_random_seed(args.seed)
    sampler_batch = args.sampler_batch

    if args.save_path is None:
        save_path1 = "-".join([args.dataset, "ProtoNet"])
        save_fields = [
                "shot{}".format(args.shot),
                "query{}".format(args.query),
                "way{}".format(args.way),
                "vway{}".format(args.validation_way),
                "epoch{}".format(args.max_epoch),
                "batch{}".format(args.sampler_batch),
                "lr{}".format(args.lr),
                "mlplr{}".format(args.mlp_lr),
                "temp{}".format(args.temperature),
                "hyp{}".format(args.hyperbolic),
                "dim{}".format(args.dim),
                "c{}".format(args.c),
                "seed{}".format(args.seed),
                "model{}".format(args.model),
            ]
        if args.model == "medical_vit":
            save_fields.append("vitmode{}".format(args.vit_train_mode))
            if args.vit_train_mode == "adapter":
                save_fields.append("vitadapter{}".format(args.vit_adapter_dim))
        save_path2 = "_".join(save_fields)
        args.save_path = save_path1 + "_" + save_path2
        ensure_path(args.save_path)
    else:
        ensure_path(args.save_path)

    if args.dataset == "ISIC":
        from dataloader.isic import ISIC as Dataset
    elif args.dataset == "SD198":
        from dataloader.sd198 import SD198 as Dataset
    elif args.dataset == "RFMiD":
        from dataloader.rfmid import RFMiD as Dataset
    else:
        raise ValueError("Non-supported Dataset.")

    trainset = Dataset("train", args)
    train_sampler = CategoriesSampler(
        trainset.label, sampler_batch, args.way, args.shot + args.query
    )
    train_loader = DataLoader(
        dataset=trainset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Meta episodes form an independently sampled stream from the training split.
    val_sampler = CategoriesSampler(
        trainset.label, 50, args.validation_way, args.shot + args.query
    )
    val_loader = DataLoader(
        dataset=trainset,
        batch_sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = ProtoNet(args)
    feature_dims = {
        "resnet12": 512,
        "resnet50": 2048,
        "densenet121": 1024,
        "medical_vit": 512,
    }
    feature_dim = feature_dims.get(args.model, 512)
    mlp = MlP(
        in_features=args.shot * feature_dim,
        hidden_features=128 if args.shot == 1 else 512,
        out_features=1,
    ).cuda()

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
    )
    optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=args.mlp_lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epoch
    )

    # Load pretrained encoder weights without a classifier head.
    model_dict = model.state_dict()
    if args.init_weights is not None:
        pretrained_dict = torch.load(args.init_weights)["params"]
        pretrained_dict = {"encoder." + k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if torch.cuda.is_available():
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = True
        model = model.cuda()

    if args.model == "medical_vit":
        trainable_params = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        total_params = sum(parameter.numel() for parameter in model.parameters())
        print("medical ViT checkpoint: {}".format(model.encoder.checkpoint))
        print(
            "medical ViT output normalization: {}".format(
                model.encoder.normalize_output
            )
        )
        print(
            "medical ViT mode: {}, trainable params: {:.3f}M / {:.3f}M".format(
                args.vit_train_mode,
                trainable_params / 1e6,
                total_params / 1e6,
            )
        )

    def save_model(name):
        if args.model == "medical_vit" and args.vit_train_mode == "adapter":
            trainable_names = {
                parameter_name
                for parameter_name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            state = {
                key: value
                for key, value in model.state_dict().items()
                if key in trainable_names
            }
            payload = {
                "params": state,
                "checkpoint_format": "trainable_only",
                "medical_vit_checkpoint": model.encoder.checkpoint,
                "vit_train_mode": args.vit_train_mode,
            }
        else:
            payload = {"params": model.state_dict()}
        torch.save(payload, osp.join(args.save_path, name + ".pth"))

    trlog = {}
    trlog["args"] = vars(args)
    trlog["train_loss"] = []
    trlog["val_loss"] = []
    trlog["train_acc"] = []
    trlog["train_acc_meta"] = []
    trlog["val_acc"] = []
    trlog["max_acc"] = 0.0
    trlog["max_acc_epoch"] = 0

    timer = Timer()
    global_count = 0
    writer = SummaryWriter(comment=args.save_path)
    path = args.save_path + '/' + 'epsilons' + '.csv'
    max_epochs = 0
    train_meta_loader_iter = iter(val_loader)

    for epoch in range(1, args.max_epoch + 1):
        tl = Averager()
        ta0 = Averager()
        ta1 = Averager()
        tam = Averager()
        vl = Averager()
        va0 = Averager()
        va1 = Averager()
        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.way
            data_shot, data_query = data[:p], data[p:]
            label_shot = torch.arange(args.way).repeat(args.shot)
            label_shot = label_shot.type(torch.cuda.LongTensor)
            label_query = torch.arange(args.way).repeat(args.query)
            label_query = label_query.type(torch.cuda.LongTensor)

            model.train()
            if args.model == "medical_vit":
                meta_model = model
            else:
                meta_model = ProtoNet(args)
                meta_model.load_state_dict(model.state_dict())
                meta_model.train().cuda()

            if medical_sdp_state is not None:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)

            feature_shot, feature_query, _ = meta_model(data_shot, data_query)

            centroid_classes = torch.unique(label_shot)
            train_prototypes = torch.stack(
                [feature_shot[torch.where(label_shot == c)[0]].mean(0) for c in centroid_classes])
            support_data = [feature_shot[torch.where(label_shot == c)] for c in centroid_classes]
            if args.shot > 1:
                support_data = torch.cat([t.view(-1) for t in support_data], dim=0).view(args.way, -1)
            else:
                support_data = torch.cat(support_data, dim=0)
            epsilons = mlp(support_data).squeeze()
            epsilons0 = torch.ones(args.way).cuda()
            epsilons0 = epsilons0 * args.epsilons_ratio
            epsilons = epsilons + epsilons0
            logits_label1, logits_label2, logits_ori = \
                logits_label(feature_query, train_prototypes, label_query, centroid_classes, epsilons, args.way)

            loss = F.cross_entropy(logits_ori, label_query)
            meta_model.zero_grad(set_to_none=True)
            if args.model == "medical_vit":
                named_trainable_params = [
                    (name, parameter)
                    for name, parameter in meta_model.named_parameters()
                    if parameter.requires_grad
                ]
                grads = torch.autograd.grad(
                    loss,
                    [parameter for _, parameter in named_trainable_params],
                    create_graph=True,
                )
                virtual_params = {
                    name: parameter - args.lr * grad
                    for (name, parameter), grad in zip(named_trainable_params, grads)
                }
            else:
                grads = torch.autograd.grad(
                    loss, meta_model.params(), create_graph=True
                )
                meta_model.update_params(lr_inner=args.lr, source_params=grads)
            del grads

            p = args.shot * args.validation_way
            label_shot_meta = torch.arange(args.validation_way).repeat(args.shot).type(torch.cuda.LongTensor)
            label_query_meta = torch.arange(args.validation_way).repeat(args.query).type(torch.cuda.LongTensor)
            centroid_classes_meta = torch.unique(label_shot_meta)
            try:
                data_meta, _ = next(train_meta_loader_iter)
            except StopIteration:
                train_meta_loader_iter = iter(val_loader)
                data_meta, _ = next(train_meta_loader_iter)
            data_meta = data_meta.cuda()
            data_shot_meta, data_query_meta = data_meta[:p], data_meta[p:]

            if args.model == "medical_vit":
                feature_shot_meta, feature_query_meta, _ = functional_call(
                    meta_model,
                    virtual_params,
                    (data_shot_meta, data_query_meta),
                )
            else:
                feature_shot_meta, feature_query_meta, _ = meta_model(
                    data_shot_meta, data_query_meta
                )

            train_prototypes_meta = torch.stack(
                [feature_shot_meta[torch.where(label_shot_meta == c)[0]].mean(0) for c in centroid_classes_meta])
            if args.hyperbolic:
                logits_ori_meta = (
                        -dist_matrix(feature_query_meta, train_prototypes_meta, c=args.c) / args.temperature)
            else:
                logits_ori_meta = -1 * torch.cdist(feature_query_meta, train_prototypes_meta)
            loss_val = F.cross_entropy(logits_ori_meta, label_query_meta)
            acc_meta = count_acc(logits_ori_meta, label_query_meta)

            optimizer_mlp.zero_grad(set_to_none=True)
            loss_val.backward()
            if medical_sdp_state is not None:
                torch.backends.cuda.enable_flash_sdp(medical_sdp_state[0])
                torch.backends.cuda.enable_mem_efficient_sdp(medical_sdp_state[1])
                torch.backends.cuda.enable_math_sdp(medical_sdp_state[2])
            optimizer_mlp.step()
            optimizer_mlp.zero_grad(set_to_none=True)

            del loss_val, logits_ori_meta, train_prototypes_meta
            del feature_shot_meta, feature_query_meta
            if args.model == "medical_vit":
                del virtual_params, named_trainable_params
            else:
                del meta_model
            del loss, logits_ori, logits_label1, logits_label2
            del feature_shot, feature_query, train_prototypes, support_data, epsilons

            feature_shot, feature_query, _ = model(data_shot, data_query)
            centroid_classes = torch.unique(label_shot)
            train_prototypes = torch.stack(
                [feature_shot[torch.where(label_shot == c)[0]].mean(0) for c in centroid_classes])
            support_data = [feature_shot[torch.where(label_shot == c)] for c in centroid_classes]
            if args.shot > 1:
                support_data = torch.cat([t.view(-1) for t in support_data], dim=0).view(args.way, -1)
            else:
                support_data = torch.cat(support_data, dim=0)
            with torch.no_grad():
                epsilons = mlp(support_data).squeeze()
                epsilons0 = torch.ones(args.way).cuda()
                epsilons0 = epsilons0 * args.epsilons_ratio
                epsilons = epsilons + epsilons0

            logits_label1, logits_label2, logits_ori = \
                logits_label(feature_query, train_prototypes, label_query, centroid_classes, epsilons, args.way)

            loss = F.cross_entropy(logits_ori, label_query)

            acc1 = count_acc(logits_label1, label_query)
            acc2 = count_acc(logits_label2, label_query)

            tl.add(loss.item())
            ta0.add(acc1)
            ta1.add(acc2)
            tam.add(acc_meta)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        model.eval()
        print(
            "best epoch {}, best val acc={:.4f}".format(
                trlog["max_acc_epoch"], trlog["max_acc"]
            )
        )
        epsilons_val = Averager()
        with torch.no_grad():
            for i, batch in enumerate(val_loader, 1):
                data, _ = [_.cuda() for _ in batch]
                p = args.shot * args.validation_way
                data_shot, data_query = data[:p], data[p:]
                label_shot = torch.arange(args.validation_way).repeat(args.shot).type(torch.cuda.LongTensor)
                label_query = torch.arange(args.validation_way).repeat(args.query).type(torch.cuda.LongTensor)
                feature_shot, feature_query, _ = model(data_shot, data_query)
                centroid_classes = torch.unique(label_shot)
                train_prototypes = torch.stack(
                    [feature_shot[torch.where(label_shot == c)[0]].mean(0) for c in centroid_classes])
                support_data = [feature_shot[torch.where(label_shot == c)] for c in centroid_classes]
                if args.shot > 1:
                    support_data = torch.cat([t.view(-1) for t in support_data], dim=0).view(args.validation_way, -1)
                else:
                    support_data = torch.cat(support_data, dim=0)
                epsilons = mlp(support_data).squeeze()
                epsilons_val.add(epsilons.mean().item())
                epsilons0 = torch.ones(args.validation_way).cuda()
                epsilons0 = epsilons0 * args.epsilons_ratio
                epsilons = epsilons + epsilons0
                logits_label1, logits_label2, logits_ori = \
                    logits_label(feature_query, train_prototypes, label_query, centroid_classes, epsilons, args.validation_way)

                loss = F.cross_entropy(logits_ori, label_query)

                acc1 = count_acc(logits_label1, label_query)
                acc2 = count_acc(logits_label2, label_query)

                vl.add(loss.item())
                va0.add(acc1)
                va1.add(acc2)


        tl = tl.item()
        ta0 = ta0.item()
        ta1 = ta1.item()
        tam = tam.item()
        vl = vl.item()
        va0 = va0.item()
        va1 = va1.item()
        epsilons_val = epsilons_val.item()
        writer.add_scalar("data/val_loss", float(vl), epoch)
        writer.add_scalar("data/val_acc0", float(va0), epoch)
        writer.add_scalar("data/val_acc1", float(va1), epoch)
        print("epoch {}, train, loss={:.4f} acc0={:.4f} acc1={:.4f} acc_meta={:.4f}".format(epoch, tl, ta0, ta1, tam))
        print("epoch {}, val, loss={:.4f} acc0={:.4f} acc1={:.4f}".format(epoch, vl, va0, va1))
        print("epsilon_mean:", epsilons_val)
        with open(path, 'a', newline='', encoding='utf-8') as csvFile:
            writer_epsilons = csv.writer(csvFile)
            writer_epsilons.writerow([epsilons_val])

        if va0 > trlog["max_acc"]:
            max_epochs = int(epoch)
            trlog["max_acc"] = va0
            trlog["max_acc_epoch"] = epoch
            save_model("max_acc")

        trlog["train_loss"].append(tl)
        trlog["train_acc"].append(ta0)
        trlog["train_acc_meta"].append(tam)
        trlog["val_loss"].append(vl)
        trlog["val_acc"].append(va0)

        torch.save(trlog, osp.join(args.save_path, "trlog"))
        save_model("epoch-last")
        if epoch % 50 == 0:
            save_model('epoch-{}'.format(epoch))
        lr_scheduler.step()
        print(
            "ETA:{}/{}".format(timer.measure(), timer.measure(epoch / args.max_epoch))
        )
    print('MAX Epoch:{}'.format(max_epochs))
    writer.close()
