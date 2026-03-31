# Eureka Visual: Automated RL Reward Engineering Pipeline

## Overview
Eureka Visual is an automated pipeline that trains locomotion policies (like Humanoid-v4 and Ant-v4) in MuJoCo using Stable Baselines3. It leverages a local Large Language Model (LLM) to write and iteratively refine Python reward functions. 

To combat "reward hacking"—where the robot achieves high scores via unnatural or broken physics—the system uses a local Vision-Language Model (CLIP) to visually evaluate rendered frames of the robot's behavior. These frames are checked against predefined text probes (e.g., "robot is tumbling"). The CLIP output is converted into a structured text report, which is then fed back to the LLM alongside quantitative training statistics to rewrite and improve the reward function.

## System Architecture
The system dynamically adapts to distinct local hardware environments:
- **M2 MacBook Air (8GB Unified Memory):** Operates under memory constraints using sequential execution (Simulation -> GC -> Vision -> GC -> LLM). Utilizes `llama3.2:3b` and `openai/clip-vit-base-patch32`.
- **RTX 4060 Laptop (8GB Dedicated VRAM):** Supports concurrent execution (MuJoCo on CPU/RAM, models on VRAM). Utilizes `llama3.1:8b-q4` and `openai/clip-vit-base-patch32`.

### The Visual Pipeline
Instead of passing raw images to the text-based LLM, the pipeline:
1. Subsamples keyframes from evaluation rollouts.
2. Passes 256x256 frames to CLIP.
3. Performs zero-shot scoring against positive/negative text probes.
4. Generates a formatted text report with explicit flags for contradictions (e.g., quantitative stats suggest high velocity, but visual stats show the robot is cartwheeling).

## Project Structure
- `config.py` - Hardware auto-detection & memory management flags.
- `eureka.py` - Main orchestrator running the Eureka loop.
- `mujoco_runner.py` - Stable Baselines3 PPO setup, AsyncVectorEnv, and frame rendering.
- `frame_extractor.py` - Subsampling logic for evaluation keyframes.
- `clip_analyzer.py` - CLIP ViT inference and text-report generation.
- `llm_feedback.py` - Ollama API integration, code generation, and reflection.
- `probes/` - Probe text dictionaries for bipedal (`humanoid.py`) and quadrupedal (`ant.py`) locomotion.
- `envs/` - Gym wrappers for injecting observation space docs and custom reward functions.

## Recent Results
The pipeline successfully completed a full 3-iteration run.
