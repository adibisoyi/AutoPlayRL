# MarioRLAgent: Autonomous RL Agent for Mario

## Overview

`RLAGENT` is a modular framework for building autonomous reinforcement learning agents for video games. The `MarioRLAgent` submodule integrates:

* **State Extraction** via YOLOv5-based object detection and custom HUD analysis
* **Reward Modeling** with memory buffers and learned reward networks
* **PPO Policy** implementation in TensorFlow/PyTorch for agent training
* **Pipeline Orchestration** through a master loop coordinating capture, detection, and action execution

## Repository Structure

```
RLAGENT/                          # Root folder for all RL agents
└── MarioRLAgent/                 # Mario-specific agent implementation
    ├── data/                     # Serialized memory, training data, etc.
    │   └── memory.pkl            # Replay/memory buffer file
    ├── models/                   # Saved model checkpoints
    │   ├── best.pt               # YOLOv5 detection weights
    ├── scripts/                  # Core scripts and utilities
    │   ├── __pycache__/          # Compiled Python files
    │   ├── agent_utils/          # Helper modules
    │   │   ├── __pycache__/
    │   │   ├── templates/        # Template configs or files
    │   │   ├── actions.py        # Keypress/action definitions
    │   │   ├── hud_analyser.py   # OCR- and strip-based HUD parsing
    │   │   ├── policy.py         # policy network
    │   │   ├── reward_memory.py  # Replay buffer implementation
    │   │   ├── reward_model.py   # Learned reward network
    │   │   ├── screen_capture.py # Frame capture and preprocessing
    │   │   ├── screen_monitor.py # Game window monitoring
    │   │   └── state_extractor.py# Object-detection-based state builder
    │   ├── data/                 # Script-specific data outputs
    │   ├── logs/                 # Training and evaluation logs
    │   ├── yolov5/               # YOLOv5 repository or integration
    │   ├── agent.py              # Entry point for training/evaluation
    │   ├── hud_monitor.py        # Standalone HUD testing pipeline
    │   ├── hud_test_pipeline.py  # HUD pipeline example script
    │   └── masterloop.py         # Main RL orchestration loop
    ├── environment.yml          # Conda env specification (if used)
    ├── requirements.txt         # pip dependencies
    └── README.md                # This overview and instructions
```

## Installation

1. **Clone**:

```bash
git clone https://github.com/adibisoyi/AutoPlayRL.git
cd RLAGENT/MarioRLAgent
```
2. **Environment**:
   - **Conda**:
```bash
conda env create -f environment.yml
conda activate rlagent
```

 - **Virtualenv + pip**:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
3. **Download YOLOv5 and pretrained weights**:
Grab the YOLOv5 repository and the `best.pt` weights from [Google Drive]((https://drive.google.com/drive/folders/1DHswa77ZItY7tJHxB_ejbhvlIkoMAFrm?usp=drive_link)).

After downloading:
1. Copy the `yolov5` folder into `scripts/yolov5/`.
2. Place `best.pt` in `models/` so the file `models/best.pt` exists.

## Usage
- **Training**:
```bash
MAX_COMBO_KEYS=2 python scripts/masterloop.py --episodes 500
```


## Contributing

PRs welcome! Please adhere to existing style, add tests, and update docs.

## License

MIT License. See [LICENSE](LICENSE) for details.
