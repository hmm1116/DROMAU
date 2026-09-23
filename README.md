# DROMAU

Clean research implementation for episodic few-shot medical image
classification. The repository contains the training and evaluation pipeline
used for ISIC, SD-198, and RFMiD, including the bi-level virtual update,
MR-Net radius prediction, Euclidean/hyperbolic prototype classification, and
multi-seed evaluation with ACC and AUC.

## Supported settings

| Dataset | Train way | Validation/test way | 1-shot query | 5-shot query |
| --- | ---: | ---: | ---: | ---: |
| ISIC | 3 | 2 | 2 | 2 |
| SD-198 | 5 | 4 | 2 | 2 |
| RFMiD | 5 | 5 | 2 | 1 |

Supported backbones are `resnet12`, `resnet50`, and `densenet121`.

## Repository layout

```text
.
|-- train.py                 # episodic training and meta optimization
|-- evaluate.py              # checkpoint evaluation (ACC and AUC)
|-- run_experiment.py        # train once, evaluate multiple test seeds
|-- run_experiment.sh        # portable experiment launcher
|-- scripts/                 # dataset/shot presets
|-- dataloader/              # dataset readers and episodic sampler
|-- models/                  # ProtoNet/DROMAU model wrapper
|-- networks/                # supported backbones and MR-Net MLP
|-- hyptorch/                # Poincare-ball operations
|-- data/                    # split files only; no medical images
└-- docs/GITHUB_RELEASE_CN.md
```

## Environment

The reference environment is:

- Python 3.9.25
- PyTorch 2.1.0+cu121, CUDA runtime 12.1, cuDNN 8.8.1
- torchvision 0.16.0
- NumPy 1.23.5
- SciPy 1.11.4
- scikit-learn 1.3.2

Create it with Conda:

```bash
conda env create -f environment.yml
conda activate dromau
```

Alternatively, install PyTorch first and then the remaining packages:

```bash
conda create -n dromau python=3.9.25 -y
conda activate dromau
pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

A CUDA-capable GPU is required by the current training and evaluation code.

## Data preparation

The repository does not redistribute medical images. Follow
[`data/README.md`](data/README.md), then set the corresponding image root:

```bash
export ISIC_IMAGE_PATH=/path/to/ISIC_2019_Training_Input
export SD198_IMAGE_PATH=/path/to/SD-198/images
export RFMID_IMAGE_PATH=/path/to/RFMiD
```

ISIC and RFMiD split CSVs are included. Add the verified SD-198 `train.csv`
and `test.csv` files to `data/SD198/split` before running SD-198.

## Training and evaluation

The presets train one model and then evaluate checkpoint `epoch-last.pth`.
The documented options below can be overridden through environment variables.

```bash
# ISIC, 2-way 1-shot evaluation
GPU=0 ISIC_IMAGE_PATH=/path/to/isic bash scripts/isic_1shot.sh

# SD-198, 4-way 5-shot evaluation
GPU=0 SD198_IMAGE_PATH=/path/to/sd198/images bash scripts/sd198_5shot.sh

# RFMiD, 5-way 5-shot evaluation
GPU=0 RFMID_IMAGE_PATH=/path/to/rfmid bash scripts/rfmid_5shot.sh
```

To evaluate an existing checkpoint without retraining, set `SKIP_TRAIN=1`,
`SAVE_PATH` to its run directory, and `CHECKPOINT` to its filename:

```bash
SKIP_TRAIN=1 SAVE_PATH=/path/to/run CHECKPOINT=epoch-last.pth \
GPU=0 RFMID_IMAGE_PATH=/path/to/rfmid bash scripts/rfmid_5shot.sh
```

You may also call `evaluate.py` directly with `--save_path` and
`--checkpoint`.

### Backbone selection

```bash
# ResNet-50
MODEL_LIST=resnet50 GPU=0 ISIC_IMAGE_PATH=/path/to/isic \
bash scripts/isic_5shot.sh
```

### Main options

| Variable | Meaning |
| --- | --- |
| `MODEL_LIST` | Backbone architecture(s) |
| `MAX_EPOCH` | Number of training epochs |
| `SAMPLER_BATCH` | Number of episodic training iterations per epoch |
| `LR_LIST` | Feature-extractor learning rate(s) |
| `MLP_LR_LIST` | MR-Net learning rate(s) |
| `C_LIST` | Poincare-ball curvature(s) |
| `TEST_EPISODES` | Number of randomly sampled evaluation episodes |

Training episodes and meta episodes are sampled independently from the same
training split; the meta stream does not read test classes or test images.

## Outputs

Each run directory contains:

- `epoch-last.pth` and `max_acc.pth`: model checkpoints;
- `trlog`: serialized training log;
- `test_results_<checkpoint>.csv`: evaluation results;
- TensorBoard logs when `tensorboardX` is installed.

Aggregated rows are appended to `results/<dataset>_results.csv` and
`results/<dataset>_summary.csv`.

## Reproducibility notes

- Keep train way and validation/test way consistent with the table above.
- Do not commit datasets, checkpoints, patient information, or local absolute
  paths.
- The exact SD-198 release split must be added before claiming reproduction of
  the reported SD-198 numbers.

## Citation

Please add the final paper citation here after publication.

## License

The project is released under the MIT License. Before making the repository
public, verify that all retained third-party code and dataset split files may
be redistributed under their original terms.
