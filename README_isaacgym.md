# IR-BPP IsaacGym Version Setup Guide

This document describes how to configure the IsaacGym environment to run the IR-BPP (Irregular Shape Bin Packing Problem) project.

---

## 1. System Requirements

- **OS**: Ubuntu 18.04 / 20.04 / 22.04
- **GPU**: NVIDIA GPU (RTX 3080 or above recommended, VRAM >= 8GB)
- **Driver**: NVIDIA Driver >= 470 (must be installed)
- **CUDA**: PyTorch bundles its own CUDA runtime; a separate CUDA Toolkit installation is usually not needed

> Note: Just make sure `nvidia-smi` displays GPU information correctly.

---

## 2. Install IsaacGym

### 2.1 Download IsaacGym

IsaacGym must be downloaded from the NVIDIA website (requires an NVIDIA developer account):

1. Visit [NVIDIA IsaacGym](https://developer.nvidia.com/isaac-gym)
2. Click "Join now" or "Download"
3. Download `IsaacGym_Preview_4_Package.tar.gz` (or the latest version)

### 2.2 Extract and install

```bash
# Extract
tar -xzf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym

# View documentation (optional)
ls docs/

# Install the IsaacGym Python package
cd python
pip install -e .
```

### 2.3 Verify installation

```bash
# Run an example to verify
cd examples
python joint_monkey.py
```

If a robotic arm simulation window appears, the installation was successful.

---

## 3. Create a Conda virtual environment

### 3.1 Create the environment

```bash
# Create a Python 3.8 environment (IsaacGym recommends 3.7-3.8)
conda create -n isaacgym python=3.8 -y
conda activate isaacgym
```

### 3.2 Install IsaacGym (automatically installs PyTorch and other dependencies)

```bash
# Navigate to the isaacgym directory
cd /path/to/isaacgym/python
pip install -e .
```

### 3.3 Verify PyTorch CUDA

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

- If output is `True`: everything is fine, skip to 3.4
- If output is `False`: reinstall PyTorch with CUDA support:

```bash
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### 3.4 Install project dependencies

```bash
# Navigate to the IR-BPP project directory
cd /home/zzx/Documents/IR-BPP

# Install dependencies
pip install -r requirements_isaacgym.txt
```

---

## 4. Install IsaacGymEnvs (optional)

If you need additional environments and utilities provided by IsaacGymEnvs:

```bash
# Clone the repository
git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
cd IsaacGymEnvs

# Install
pip install -e .
```

---

## 5. Run training

### 5.1 Basic run

```bash
conda activate isaacgym
cd /home/zzx/Documents/IR-BPP

# Run IsaacGym version training
python main_isaacgym.py --custom experiment_name
```

### 5.2 Common command-line arguments

```bash
# Specify GPU
python main_isaacgym.py --device 0 --custom exp1

# Adjust batch size
python main_isaacgym.py --batch-size 256 --custom exp2

# Adjust learning rate
python main_isaacgym.py --learning-rate 3e-4 --custom exp3

# Full example
python main_isaacgym.py \
    --device 0 \
    --batch-size 256 \
    --learning-rate 3e-4 \
    --custom my_experiment
```

---

## 6. Key parameter configuration

Key parameters are set in the **IsaacGym Specific Configuration** section of `main_isaacgym.py`:

### 6.1 Parameter descriptions

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `num_processes` | main_isaacgym.py | 256 | Number of parallel environments; more = faster training (limited by VRAM) |
| `memory_capacity` | main_isaacgym.py | 10000 | Replay buffer size; recommend `num_processes * 100` |
| `learning_rate` | main_isaacgym.py | 1e-4 | Learning rate |
| `batch_size` | arguments.py | 64 | Batch size; recommend 256-512 |
| `target_update` | arguments.py | 1000 | Target network update interval |
| `learn_start` | arguments.py | 500 | Steps before learning begins |
| `headless` | main_isaacgym.py | False | **True = no rendering (faster training), False = with rendering** |

### 6.2 Modifying parameters in main_isaacgym.py

Open `main_isaacgym.py` and find the `# ==================== IsaacGym Specific Configuration ====================` section:

```python
# ==================== IsaacGym Specific Configuration ====================
# Number of parallel environments (adjust based on GPU VRAM; RTX 3090 can use 1024-2048)
args.num_processes = 1024

# Replay buffer size (recommend num_processes * 100)
args.memory_capacity = 100000

# Learning rate (increase for large batches)
args.learning_rate = 3e-4

# Batch size (increase for large num_processes)
args.batch_size = 256

# Target network update frequency (reduce to speed up learning)
args.target_update = 500

# Steps before learning begins (reduce for large num_processes)
args.learn_start = 200
```

### 6.3 Disable rendering (important!)

**Always disable rendering during training to improve speed:**

```python
env = PackingGame(
    args=args,
    num_envs=args.num_processes,
    sim_device=args.device.index if args.device.type == 'cuda' else 0,
    headless=True  # <- set to True to disable rendering
)
```

---

## 7. Recommended configurations

### 7.1 Fast training (RTX 3090 / 4090)

```python
args.num_processes = 1024      # large-scale parallelism
args.memory_capacity = 100000  # large replay buffer
args.batch_size = 512          # large batch
args.learning_rate = 3e-4      # higher learning rate
args.target_update = 500       # more frequent target network updates
args.learn_start = 200         # start learning sooner
headless = True                # disable rendering
```

### 7.2 Stable training (smaller GPU VRAM)

```python
args.num_processes = 256       # moderate parallelism
args.memory_capacity = 50000   # moderate replay buffer
args.batch_size = 128          # moderate batch
args.learning_rate = 1e-4      # standard learning rate
args.target_update = 1000      # standard update frequency
args.learn_start = 500         # standard warmup
headless = True                # disable rendering
```

---

## 8. Run evaluation

Use `test_isaacgym.py` to evaluate a trained model.

### 8.1 Basic usage

```bash
conda activate isaacgym
cd /home/zzx/Documents/IR-BPP

# Specify model path and run test
python test_isaacgym.py --model ./logs/experiment/xxx/checkpoint.pt
```

### 8.2 Test parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | (required) | Path to the trained model file (.pt) |
| `--test-episodes` | 100 | Number of test episodes |
| `--test-envs` | 16 | Number of parallel environments (speeds up testing) |
| `--test-render` | off | Add this flag to enable visualization |

### 8.3 Common test commands

```bash
# Quick test (few episodes, no rendering)
python test_isaacgym.py --model ./logs/experiment/xxx/checkpoint.pt \
    --test-episodes 50 --test-envs 32

# Full test (more episodes, more reliable statistics)
python test_isaacgym.py --model ./logs/experiment/xxx/checkpoint.pt \
    --test-episodes 500 --test-envs 64

# Visual test (enable rendering to observe placement)
python test_isaacgym.py --model ./logs/experiment/xxx/checkpoint.pt \
    --test-episodes 10 --test-envs 4 --test-render
```

### 8.4 Test output

After testing, the following statistics are printed:

- **Reward statistics**: mean, standard deviation
- **Episode length statistics**: number of items successfully packed per episode
- **Space utilization (Ratio)**: mean, standard deviation, min, max

Example output:

```
============================================================
   Test Results
============================================================
  Completed episodes: 100

  Reward statistics:
    - Mean: 3.2451
    - Std:  1.1234

  Episode length statistics:
    - Mean: 5.60
    - Std:  2.13

  Space utilization (Ratio) statistics:
    - Mean: 0.3245 (32.45%)
    - Std:  0.0812
    - Min:  0.1523 (15.23%)
    - Max:  0.5134 (51.34%)
============================================================
```

### 8.5 Notes

- Test parameters (`scale`, `bin_dimension`, `ZRotNum`) **must match those used during training**; otherwise the model output will be incorrect
- These parameters are set on lines 31-33 of `test_isaacgym.py`; they must correspond to the values in `main_isaacgym.py`
- Model files are saved under `./logs/experiment/<experiment_name>/` with the filename format `checkpoint<timestamp>.pt`

---

## 9. View training logs

Training logs are saved under `./logs/runs/`. View them with TensorBoard:

```bash
tensorboard --logdir=./logs/runs/
```

Then open `http://localhost:6006` in a browser.

---

## 10. Common issues

### Q1: ImportError: libpython3.8.so.1.0

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### Q2: CUDA out of memory

Reduce `num_processes` or `batch_size`:

```python
args.num_processes = 128  # reduce parallelism
args.batch_size = 64      # reduce batch size
```

### Q3: IsaacGym cannot find GPU

```bash
# Check CUDA
nvidia-smi

# Set visible CUDA devices
export CUDA_VISIBLE_DEVICES=0
```

### Q4: Training is slow

1. Ensure `headless=True`
2. Increase `num_processes`
3. Check that the model is not running on CPU

---

## 11. Project structure

```
IR-BPP/
├── main_isaacgym.py           # IsaacGym version training entry point
├── test_isaacgym.py           # IsaacGym version test script
├── trainer_isaacgym.py        # IsaacGym version trainer
├── arguments.py               # Command-line arguments
├── agent.py                   # DQN Agent
├── requirements_isaacgym.txt  # IsaacGym version dependencies
├── environment/
│   └── physics0/
│       ├── binPhy_isaacgym.py     # IsaacGym environment
│       ├── Interface_isaacgym.py  # IsaacGym physics interface
│       ├── IRcreator_isaacgym.py  # IsaacGym item creator
│       ├── binPhy.py              # PyBullet environment (original)
│       └── Interface.py           # PyBullet physics interface (original)
├── dataset/                   # Dataset
└── logs/
    ├── runs/                  # TensorBoard logs
    └── experiment/            # Model checkpoints
```

---

## 12. References

- [IsaacGym Official Documentation](https://developer.nvidia.com/isaac-gym)
- [IsaacGym GitHub Examples](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
