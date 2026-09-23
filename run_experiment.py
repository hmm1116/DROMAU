import argparse
import csv
import os
import subprocess
import sys


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument(
        "--model", type=str, default="resnet12",
        choices=["resnet12", "resnet50", "densenet121", "medical_vit"]
    )
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=5)
    parser.add_argument("--way", type=int, default=3)
    parser.add_argument("--validation_way", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--mlp_lr", type=float, default=1e-2)
    parser.add_argument(
        "--dataset", type=str, default="ISIC", choices=["ISIC", "SD198", "RFMiD"]
    )
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--hyperbolic", action="store_true", default=False)
    parser.add_argument("--c", type=float, default=None)
    parser.add_argument("--init_weights", type=str, default=None)
    parser.add_argument("--gpu", default="0,1,2,3")
    parser.add_argument("--train_c", action="store_true", default=False)
    parser.add_argument("--train_x", action="store_true", default=False)
    parser.add_argument("--not-riemannian", action="store_true")
    parser.add_argument("--sampler_batch", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_seeds", type=str, default="0")
    parser.add_argument("--test_episodes", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, default="epoch-last.pth")
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
    parser.add_argument("--results_csv", type=str, default="./results/ablation_results.csv")
    parser.add_argument("--summary_csv", type=str, default=None)
    parser.add_argument("--skip_train", action="store_true", default=False)
    return parser


def default_save_path(args):
    fields = [
        ("shot", args.shot),
        ("query", args.query),
        ("way", args.way),
        ("vway", args.validation_way),
        ("epoch", args.max_epoch),
        ("batch", args.sampler_batch),
        ("lr", args.lr),
        ("mlplr", args.mlp_lr),
        ("hyp", args.hyperbolic),
        ("c", args.c),
        ("seed", args.seed),
        ("model", args.model),
    ]
    if args.model == "medical_vit":
        fields.append(("vitmode", args.vit_train_mode))
        if args.vit_train_mode == "adapter":
            fields.append(("vitadapter", args.vit_adapter_dim))
    return "{}-ProtoNet_{}".format(
        args.dataset, "_".join("{}{}".format(k, v) for k, v in fields)
    )


def add_common_args(cmd, args, include_train_only):
    values = {
        "--dataset": args.dataset,
        "--gpu": args.gpu,
        "--query": args.query,
        "--way": args.way,
        "--validation_way": args.validation_way,
        "--shot": args.shot,
        "--lr": args.lr,
        "--mlp_lr": args.mlp_lr,
        "--model": args.model,
        "--c": args.c,
        "--sampler_batch": args.sampler_batch,
        "--num_workers": args.num_workers,
    }
    if include_train_only:
        values["--max_epoch"] = args.max_epoch
    else:
        values["--test_episodes"] = args.test_episodes

    for key, value in values.items():
        cmd.extend([key, str(value)])

    if args.init_weights is not None:
        cmd.extend(["--init_weights", args.init_weights])
    if args.medical_vit_dir is not None:
        cmd.extend(["--medical_vit_dir", args.medical_vit_dir])
    if args.model == "medical_vit":
        cmd.extend(["--vit_train_mode", args.vit_train_mode])
        cmd.extend(["--vit_adapter_dim", str(args.vit_adapter_dim)])
    if args.isic_image_path is not None:
        cmd.extend(["--isic_image_path", args.isic_image_path])
    if args.isic_split_path is not None:
        cmd.extend(["--isic_split_path", args.isic_split_path])
    if args.sd198_image_path is not None:
        cmd.extend(["--sd198_image_path", args.sd198_image_path])
    if args.sd198_split_path is not None:
        cmd.extend(["--sd198_split_path", args.sd198_split_path])
    if args.rfmid_image_path is not None:
        cmd.extend(["--rfmid_image_path", args.rfmid_image_path])
    if args.rfmid_split_path is not None:
        cmd.extend(["--rfmid_split_path", args.rfmid_split_path])
    if args.hyperbolic:
        cmd.append("--hyperbolic")
    if args.train_c:
        cmd.append("--train_c")
    if args.train_x:
        cmd.append("--train_x")
    if args.not_riemannian:
        cmd.append("--not-riemannian")


def run_train(args, save_path):
    cmd = [sys.executable, "train.py"]
    add_common_args(cmd, args, include_train_only=True)
    cmd.extend(["--save_path", save_path])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    subprocess.run(cmd, check=True)


def run_test(args, save_path, seed):
    cmd = [sys.executable, "evaluate.py"]
    add_common_args(cmd, args, include_train_only=False)
    cmd.extend(["--save_path", save_path, "--checkpoint", args.checkpoint, "--seed", str(seed)])
    try:
        completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        print("test command failed: {}".format(" ".join(cmd)))
        if exc.stdout:
            print("test stdout:\n{}".format(exc.stdout))
        if exc.stderr:
            print("test stderr:\n{}".format(exc.stderr))
        raise
    print(completed.stdout, end="")
    result_lines = [
        line for line in completed.stdout.splitlines()
        if line.count(",") == 4 and not line.startswith("seed,")
    ]
    if not result_lines:
        raise RuntimeError("Could not parse test output for seed {}".format(seed))
    seed_s, acc, acc_ci, auc, auc_ci = result_lines[-1].split(",")
    return {
        "seed": int(seed_s),
        "acc": float(acc),
        "acc_ci": float(acc_ci),
        "auc": float(auc),
        "auc_ci": float(auc_ci),
    }


def result_row(args, save_path, row_type, test_seed, acc, acc_ci, auc, auc_ci):
    return {
        "row_type": row_type,
        "dataset": args.dataset,
        "model": args.model,
        "shot": args.shot,
        "query": args.query,
        "way": args.way,
        "validation_way": args.validation_way,
        "lr": args.lr,
        "mlp_lr": args.mlp_lr,
        "c": args.c,
        "medical_vit_dir": args.medical_vit_dir or "auto",
        "vit_train_mode": args.vit_train_mode if args.model == "medical_vit" else "",
        "vit_adapter_dim": (
            args.vit_adapter_dim
            if args.model == "medical_vit" and args.vit_train_mode == "adapter"
            else ""
        ),
        "train_seed": args.seed,
        "test_seed": test_seed,
        "max_epoch": args.max_epoch,
        "sampler_batch": args.sampler_batch,
        "checkpoint": args.checkpoint,
        "acc": acc,
        "acc_ci": acc_ci,
        "auc": auc,
        "auc_ci": auc_ci,
        "save_path": os.path.abspath(save_path),
    }


def write_results(csv_path, rows, append):
    if not csv_path:
        return
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fieldnames = [
        "row_type",
        "dataset",
        "model",
        "shot",
        "query",
        "way",
        "validation_way",
        "lr",
        "mlp_lr",
        "c",
        "medical_vit_dir",
        "vit_train_mode",
        "vit_adapter_dim",
        "train_seed",
        "test_seed",
        "max_epoch",
        "sampler_batch",
        "checkpoint",
        "acc",
        "acc_ci",
        "auc",
        "auc_ci",
        "save_path",
    ]
    write_header = not append or not os.path.exists(csv_path)
    mode = "a" if append else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def parse_seed_list(seed_text):
    return [int(seed) for seed in seed_text.replace(",", " ").split() if seed]


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.c is None:
        args.c = 1e-3 if args.shot == 1 else 5e-3
    save_path = args.save_path or default_save_path(args)
    test_seeds = parse_seed_list(args.test_seeds)
    if args.summary_csv is None and args.results_csv:
        root, ext = os.path.splitext(args.results_csv)
        args.summary_csv = "{}_summary{}".format(root, ext or ".csv")

    print("save_path={}".format(os.path.abspath(save_path)))
    print("results_csv={}".format(os.path.abspath(args.results_csv)))
    if args.summary_csv:
        print("summary_csv={}".format(os.path.abspath(args.summary_csv)))
    if not args.skip_train:
        run_train(args, save_path)

    results = [run_test(args, save_path, seed) for seed in test_seeds]

    print("seed,acc,acc_ci,auc,auc_ci")
    for result in results:
        print("{seed},{acc:.4f},{acc_ci:.4f},{auc:.4f},{auc_ci:.4f}".format(**result))

    acc_values = [result["acc"] for result in results]
    auc_values = [result["auc"] for result in results]
    mean_acc = sum(acc_values) / len(acc_values)
    mean_auc = sum(auc_values) / len(auc_values)
    print("mean,{:.4f},,{:.4f},".format(mean_acc, mean_auc))

    rows = [
        result_row(
            args,
            save_path,
            "seed",
            result["seed"],
            "{:.4f}".format(result["acc"]),
            "{:.4f}".format(result["acc_ci"]),
            "{:.4f}".format(result["auc"]),
            "{:.4f}".format(result["auc_ci"]),
        )
        for result in results
    ]
    rows.append(
        result_row(
            args,
            save_path,
            "mean",
            "mean",
            "{:.4f}".format(mean_acc),
            "",
            "{:.4f}".format(mean_auc),
            "",
        )
    )

    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    per_run_csv = os.path.join(save_path, "test_results_{}.csv".format(checkpoint_name))
    write_results(per_run_csv, rows, append=False)
    write_results(args.results_csv, rows, append=True)
    write_results(args.summary_csv, [rows[-1]], append=True)
    print("per_run_csv={}".format(os.path.abspath(per_run_csv)))
    print("results_csv={}".format(os.path.abspath(args.results_csv)))
    if args.summary_csv:
        print("summary_csv={}".format(os.path.abspath(args.summary_csv)))
