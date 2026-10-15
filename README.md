# DROMAU

Clean research implementation for episodic few-shot medical image
classification. The repository contains the training and evaluation pipeline
used for ISIC, SD-198, and RFMiD, including the bi-level virtual update,
MR-Net radius prediction, Euclidean/hyperbolic prototype classification, and
multi-seed evaluation with ACC and AUC.

## Supported settings

| Dataset | Train/meta way | Validation/test way | 1-shot query | 5-shot query |
| --- | ---: | ---: | ---: | ---: |
| ISIC | 3 | 2 | 2 | 2 |
| SD-198 | 5 | 4 | 2 | 2 |
| RFMiD | 5 | 5 | 1 | 1 |

Supported backbones are `resnet12`, `resnet50`, `densenet121`, and `ViT-B/16`.

## Repository layout

```text
.
|-- train.py                 # episodic training and meta optimization
|-- evaluate.py              # checkpoint evaluation (ACC and AUC)
|-- run_experiment.py        # train once, evaluate multiple test seeds
|-- run_experiment.sh        # five training runs, five evaluations per model
|-- scripts/                 # dataset/shot presets
|-- dataloader/              # dataset readers and episodic sampler
|-- models/                  # ProtoNet/DROMAU model wrapper
|-- networks/                # supported backbones and MR-Net MLP
|-- hyptorch/                # Poincare-ball operations
└-- data/                    # no medical images
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

The repository does not redistribute medical images. Set the corresponding image root:

```bash
export ISIC_IMAGE_PATH=/path/to/ISIC_2019_Training_Input
export SD198_IMAGE_PATH=/path/to/SD-198/images
export RFMID_IMAGE_PATH=/path/to/RFMiD
```

## Training and evaluation

For each configuration, the presets train five independent models and evaluate
each model's `max_acc.pth` with five evaluation seeds (2,000 episodes per seed).
The documented options below can be overridden through environment variables.

```bash
# ISIC, 2-way 1-shot evaluation
GPU=0 ISIC_IMAGE_PATH=/path/to/isic bash scripts/isic_1shot.sh

# SD-198, 4-way 5-shot evaluation
GPU=0 SD198_IMAGE_PATH=/path/to/sd198/images bash scripts/sd198_5shot.sh

# RFMiD, 5-way 5-shot evaluation
GPU=0 RFMID_IMAGE_PATH=/path/to/rfmid bash scripts/rfmid_5shot.sh
```

### Random seeds

`TRAIN_SEEDS` controls independent model training. `TEST_SEEDS` controls repeated
evaluation of each trained model. Both lists are space-separated. A nonzero seed
fixes Python, NumPy, and PyTorch random states; `0` leaves them unfixed.

```bash
# One fixed training run, followed by the default five evaluations
TRAIN_SEEDS="332" GPU=0 RFMID_IMAGE_PATH=/path/to/rfmid \
bash scripts/rfmid_5shot.sh

# One unfixed training run; evaluation still uses the fixed default test seeds
TRAIN_SEEDS="0" GPU=0 RFMID_IMAGE_PATH=/path/to/rfmid \
bash scripts/rfmid_5shot.sh
```

`run_experiment.py` handles one training run at a time;
the shell launcher supplies each training seed through `--seed` and the
evaluation list through `--test_seeds`. Default checkpoint directory names
contain the training seed. With an explicit `SAVE_PATH` and multiple training
seeds, it is used as a parent directory containing `seed1`, `seed42`, and so on;
with one training seed it is the exact run directory. Use one hyperparameter
configuration when supplying an explicit `SAVE_PATH`.

To evaluate an existing checkpoint without retraining, set `SKIP_TRAIN=1`,
`TRAIN_SEEDS` to that model's training seed, `SAVE_PATH` to its run directory,
and `CHECKPOINT` to its filename:

```bash
SKIP_TRAIN=1 TRAIN_SEEDS="332" SAVE_PATH=/path/to/run CHECKPOINT=max_acc.pth \
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
| `TRAIN_SEEDS` | Seeds for independent training runs |
| `TEST_SEEDS` | Evaluation seeds reused for each trained model |
| `MAX_EPOCH` | Number of training epochs |
| `SAMPLER_BATCH` | Number of episodic training iterations per epoch |
| `LR_LIST` | Feature-extractor learning rate(s) |
| `MLP_LR_LIST` | MR-Net learning rate(s) |
| `C_LIST` | Poincare-ball curvature(s) |
| `TEST_EPISODES` | Number of randomly sampled evaluation episodes |


## Outputs

Each run directory contains:

- `epoch-last.pth` and `max_acc.pth`: feature-extractor and MR-Net checkpoints,
  including the epoch, validation accuracies, and training arguments;
- `trlog`: serialized training log;
- `epsilons.csv`: one row per epoch with mean, minimum, and maximum radii for
  training and held-out validation separately. 
- `test_results_<checkpoint>.csv`: evaluation results;
- TensorBoard logs when `tensorboardX` is installed.

Aggregated rows are appended to `results/<dataset>_results.csv` and
`results/<dataset>_summary.csv`.
For each trained model, the results file contains five `row_type=seed` rows
and one `row_type=mean` row averaging its five evaluations. The summary file
contains one such mean row per training run. Main-table means and population
standard deviations are calculated across these five training-run means.
The `acc_ci` and `auc_ci` fields are episode-level 95% CI half-widths for one
evaluation.


## Citation

If you find this repository useful in your research, please cite the
corresponding paper. The BibTeX entry will be added here once the final
bibliographic information becomes available.

## License

The original source code in this repository is released under the
[MIT License](LICENSE).

This license applies only to code authored for this project. Third-party
libraries and any files carrying their own license notices remain subject to
their respective terms. Datasets and pretrained model weights are not covered
by this license and must be obtained from their official sources in accordance
with the corresponding licenses and terms of use.
