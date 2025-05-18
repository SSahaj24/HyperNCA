
---

<div align="center">

# HyperNCA revisited: Exploring Multi-environment and Novelty-based training

[![Paper](https://img.shields.io/badge/paper-arxiv.2204.11674-B31B1B.svg)](https://arxiv.org/abs/2204.11674)
</div>

---

## Branches
- **main**: Standard HyperNCA with multi-environment and regularization extensions.
- **metamorphosis**: Implements metamorphosis neural networks with morphing weights logic, with setup instructions included.
- **novelty**: Implements novelty-based fitness using Locality Sensitive Hashing (LSH) for efficient exploration (see below).

This readme just describes the main branch; switch to the other branch for relevant instructions.

---

This repository is a fork of the original HyperNCA paper, extended for CS6024 Course Project. We explore multi-environment training and novelty-based approaches to enhance the original HyperNCA architecture.

![](images/main.png)

## Overview

This repository contains code for growing neural networks using the HyperNCA method, with extensions for:
- Noise and Dropout regularization
- Multi-environment training capabilities
- Novelty-based training approaches

## Installation

These steps ensure that the BulletAnt environments work correctly. Setting this up was non-trivial and took significant time to figure out, so please follow carefully.

```bash
# Clone the repository
git clone https://github.com/SSahaj24/HyperNCA

# Navigate to the project directory
cd HyperNCA

# Install Python 3.10 (required)
# You can use pyenv or your system's package manager

# Example using pyenv:
pyenv install 3.10.0
pyenv virtualenv 3.10.0 venv
pyenv activate venv

# OR, if python3.10 is already installed:
# python3.10 -m venv venv
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# ⚠️ IMPORTANT: Patch the BulletAnt environments manually
# This step is required and was discovered after much trial and error

# Move XML files to pybullet_data/mjcf
mv -f bullet\ ants/*.xml venv/lib/python3.10/site-packages/pybullet_data/mjcf/

# Move Python environment files to pybullet_envs
mv -f bullet\ ants/*.py venv/lib/python3.10/site-packages/pybullet_envs/
```

## Usage

### Training

To train an agent using HyperNCA, use the `train_NCA.py` script:

```bash
# Example: Train HyperNCA on Lunar Lander
python train_NCA.py --environment LunarLander-v2
```

For all available training options, run:
```bash
python train_NCA.py --help
```

Key training parameters include:
- `--environment`: Any state-vector OpenAI Gym or pyBullet environment
- `--generations`: Number of ES generations
- `--popsize`: Population size
- `--NCA_steps`: Number of NCA steps
- `--NCA_dimension`: NCA dimension (3 uses a single 3D seed and 3DConvs)
- `--NCA_channels`: Number of NCA channels
- `--seed_type`: Seed type (single_seed, randomU2: [-1,1])

### Regularization Parameters
- `--cell_noise`: Gaussian noise standard deviation applied during NCA updates
- `--policy_noise`: Gaussian noise standard deviation applied before policy evaluation
- `--nca_dropout_rate`: Cell dropout probability during NCA updates
- `--network_dropout_rate`: Cell dropout probability just before policy evaluation
- `--layerwise_dropout_rate`: Fraction of weights to dropout in each policy network layer independently

### Weights & Biases Integration
- `--use_wandb`: Enable Weights & Biases for experiment tracking
- `--wandb_log_interval`: Log to wandb every N generations
- `--run_sweep`: Run a wandb sweep for hyperparameter optimization
- `--model_id`: ID of the saved model to use as base configuration for sweep
- `--trial_count`: Number of trials to run in the sweep

### Evaluation

To evaluate a trained model:

```bash
python fitness_functions.py --id <run_id>
```

### Reproducing Results

The following model IDs are available for evaluation:

| Id            | Environment    | Substrate |
| ------------- |:-------------:| ------:|
| 1645447353    | Lander        | Random 5 layers |
| 1646940683    | Lander        | Single 4 layers |
| 1647084085    | Quadruped     | Single 4 layers |
| 1645360631    | Quadruped     | Random 3 layers |
| 1645605120    | Quadruped     | Random 30 layers |

Model configuration files can be found in the `saved_models` directory.

## Metamorphosis Neural Networks

The metamorphosis branch contains code for training metamorphosis neural networks. The implementation uses the same NCA architecture but with modified logic for morphing weights in the RL agent.

To evaluate the reported metamorphosis model:
```bash
git checkout metamorphosis
python fitness_functions.py --id 1644785913
```

However, within the scope of the CS6024 project, we were unable to get the metamorphosis training working.

## Acknowledgments

This work builds upon the original HyperNCA paper by Najarro et al. (2022). We extend our sincere gratitude to Elias Najarro, Shyam Sudhakaran, Claire Glanois, and Sebastian Risi for their groundbreaking work on growing developmental networks with neural cellular automata. Their original implementation and insights have been invaluable to our research.

## Citation

If you use this code in your research, please cite the original paper which this fork builds on:

```bibtex
@inproceedings{najarro2022hypernca,
  title={HyperNCA: Growing Developmental Networks with Neural Cellular Automata},
  author={Najarro, Elias and Sudhakaran, Shyam and Glanois, Claire and Risi, Sebastian},
  doi = {10.48550/ARXIV.2204.11674},
  url = {https://arxiv.org/abs/2204.11674},
  booktitle={From Cells to Societies: Collective Learning across Scales},
  year={2022}
}
```