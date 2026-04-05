# Spleen Segmentaion with SAM3 and MedSAM3 

## Setup

```bash
git clone https://github.com/jon-vx/team-SAMv3-impact.git
```

## Download Weights

### SAM3 Base Weights

```bash
# TODO: add source URL / instructions
```

### MedSAM3 LoRA Weights

```bash
# TODO: add source URL / instructions
```

## Usage

### Inference

```bash
# TODO: training command and config
```

### Training

```bash
# TODO: training command and config
```

### Visualization

```bash
# TODO: plotting / evaluation commands
```

### Web Interface

```bash
# TODO: server launch command
```

## Project Structure

```
.
├── checkpoints/           # model weights (not tracked in git)
├── datasets/              # training/eval data (not tracked in git)
├── runs/                  # training logs (not tracked in git)
├── src/
│   └── impact_team_2/
│       ├── train/         # training and model loading
│       ├── vendor/        # vendored dependencies (MedSAM3)
│       ├── visual/        # plotting and evaluation
│       └── web/           # web interface
└── pyproject.toml
```

## Acknowledgments

- [MedSAM3](https://github.com/Joey-S-Liu/MedSAM3) — Liu et al., 2025
- [SAM3](https://github.com/facebookresearch/sam3) — Meta AI
