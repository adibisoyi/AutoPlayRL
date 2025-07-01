# MarioRLAgent: Autonomous RL Agent for Mario

## Overview

A modular, end-to-end proof-of-concept for autonomous game playing agents. Extendable to any title, currently demoed on Super Mario Bros.

## Core Components

**Real-Time State Extraction**  
  - YOLOv5-based object detection (scripts/agent_utils/state_extractor.py)  
  - Custom HUD parsing via OCR & strip analysis (hud_analyser.py / hud_monitor.py)  

**Configurable Reward Modeling**  
  - Heuristic reward networks & memory buffers (reward_model.py / reward_memory.py)  
  - Template-driven reward definitions for rapid experimentation  

**Epsilon-Greedy Policy Engine**  
  - Dynamic ε-greedy exploration & softmax sampling (policy.py)  
  - Clear interface for swapping in advanced RL algorithms  

**High-Performance Pipeline Orchestration**  
  - Master loop (masterloop.py) coordinating capture → detection → reward → action  
  - Cross-process shared-memory IPC queues for sub-5 ms handoff  

## Architecture Diagram
![image](https://github.com/user-attachments/assets/49ed442e-aa25-4ab0-b85a-d55bd659bacc)


## Repository Structure
```
    AutoPlayRL/                 # Mario-specific agent implementation
    ├── data/                     # Serialized memory, training data, etc.
    │   └── memory.pkl            # Replay/memory buffer file
    ├── models/                   # Saved model checkpoints
    │   ├── best.pt               # YOLOv5 detection weights
    ├── scripts/                  # Core scripts and utilities
    │   ├── agent_utils/          # Helper modules
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

This repository does not include the YOLOv5 code or the trained `best.pt` weights because of their size.  

Grab the YOLOv5 repository and the `best.pt` weights from [Google Drive](https://drive.google.com/drive/folders/1DHswa77ZItY7tJHxB_ejbhvlIkoMAFrm?usp=drive_link).

After downloading:
- Copy the `yolov5` folder into `scripts/yolov5/`.
- Place `best.pt` in `models/` so the file `models/best.pt` exists.

## Platform Requirements

This project currently targets **macOS**. Several utilities depend on Apple's Quartz framework through the `pyobjc` bindings:

- `scripts/agent_utils/screen_capture.py` locates the emulator window and grabs frames using Quartz APIs.
- `scripts/hud_monitor.py` captures HUD strips via Quartz-powered screenshots.

These pieces will not function on Windows or Linux without replacement. Keyboard input is handled with `pynput` and is cross‑platform, but the capture logic would need an alternative such as `mss` or `pyautogui` for other operating systems. Contributions adding non‑macOS backends are welcome.

All required Python packages are listed in `requirements.txt` or `environment.yml`.

## Usage
- **Training**:
```bash
MAX_COMBO_KEYS=2 python scripts/masterloop.py --episodes 500
```
> **Tip:** To allow more simultaneous key-press combinations, bump up `MAX_COMBO_KEYS` (e.g. `MAX_COMBO_KEYS=3 python scripts/masterloop.py …`).

## Research Proof-of-Concept
  - Modularity: Swap in PPO, SAC or custom policies by adhering to policy.py interface.
  -	Performance: Shared-memory IPC reduces inter-process latency by over 60%.
  -	Extendibility: Plug in new detectors (e.g. transformers) or reward schemes with minimal glue code.


## Contributing

-	Fork the repo
-	Create a topic branch
-	Add tests & update documentation
-	Submit a pull request

## License

MIT License. See [LICENSE](LICENSE) for details.
