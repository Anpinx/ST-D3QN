# ST-D3QN: UAV Path Planning with Enhanced Deep Reinforcement Learning

This project is an implementation of the **ST-D3QN** algorithm, which is an enhanced Deep Reinforcement Learning (DRL) framework for Unmanned Aerial Vehicle (UAV) path planning in ultra-low altitude environments. The algorithm combines the Dueling Double Deep Q-Network (D3QN) with the A* (A-star) algorithm to improve UAV trajectory optimization, obstacle avoidance, and reward optimization in complex environments.
![ST-D3QN](https://github.com/user-attachments/assets/ca602541-6fce-4b3f-af9b-0e976c38d59a)


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Environment Overview](#environment-overview)
- [Code Structure](#code-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Testing the Model](#testing-the-model)
- [Results](#results)
- [References](#references)

## Introduction

UAVs operating in ultra-low altitude environments face numerous challenges due to dense static and dynamic obstacles. Traditional path planning algorithms often struggle in such complex and dynamic settings. The ST-D3QN algorithm aims to address these challenges by:

- Enhancing the D3QN algorithm with sub-target guidance from the A* algorithm.
- Introducing an improved greedy strategy to prevent local optima entrapment.
- Optimizing sparse rewards to improve convergence rates during training.

This project provides a simulation environment and implementation of the ST-D3QN algorithm for UAV path planning, enabling researchers and practitioners to reproduce and build upon the work described in the associated research article.

## Features

- **Dynamic Environment Simulation**: An 11x11 grid-based environment with static and dynamic obstacles.
- **UAV Agent**: Simulated UAV capable of sensing, decision-making, and navigation.
- **ST-D3QN Algorithm**: Combines D3QN with A* sub-target guidance and an improved greedy strategy.
- **Real-time Visualization**: Visual representation of the UAV's path planning and obstacle avoidance.
- **Extensible Framework**: Modular code structure for easy adaptation and extension.

## Environment Overview

- **Grid Size**: 11x11 cells.
- **Static Obstacles**: Represented as black squares on the grid.
- **Dynamic Obstacles**: Represented as red squares; they move randomly in the environment.
- **UAV**: Represented as a green square; starts at position `[0, 0]`.
- **Goal Position**: Represented as a yellow square; located at `[9, 10]`.

## Code Structure

- `Env_ST_D3QN.py`: Defines the simulation environment, including the UAV, obstacles, and reward functions.
- `ST_D3QN.py`: Implements the ST-D3QN algorithm with neural network architectures and training methods.
- `A_star.py`: Provides the A* algorithm for calculating optimal paths and sub-targets.
- `Replaybuffer.py`: Implements the experience replay buffer for storing and sampling experiences.
- `utils.py`: Contains utility functions, such as directory creation.
- `train_ST_D3QN.py`: Script to train the ST-D3QN model.
- `test_ST_D3QN.py`: Script to test the trained model.
- `checkpoints/`: Directory to save and load model checkpoints.
- `data/`: Directory to save training data and logs.

## Dependencies

Ensure that you have the following dependencies installed:

- Python 3.8 or higher
- `numpy`
- `matplotlib`
- `torch` (PyTorch)
- `argparse`

You can install the required packages using `pip`:

```bash
pip install numpy matplotlib torch argparse
```

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Anpinx/ST-D3QN.git
   cd ST-D3QN-UAV-Path-Planning
   ```

2. **Install Dependencies**

   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

   *(Note: Ensure that `requirements.txt` contains all necessary dependencies.)*

3. **Directory Setup**

   Ensure the following directory structure exists:

   ```
   checkpoints/ST_D3QN/
   checkpoints/ST_D3QN/Q_eval/
   checkpoints/ST_D3QN/Q_target/
   data/ST_D3QN/
   ```

   You can create these directories using the provided `utils.py` utility or manually.

## Usage

### Training the Model

To train the ST-D3QN model, run the `train_ST_D3QN.py` script:

```bash
python train_ST_D3QN.py
```

**Command-line Arguments:**

- `--max_episodes`: (Optional) Maximum number of training episodes (default: 1000).
- `--ckpt_dir`: (Optional) Directory to save model checkpoints (default: `./checkpoints/ST_D3QN/`).
- `--ckpt_dir_pkl`: (Optional) Directory to save training data logs (default: `./data/ST_D3QN/`).

Example:

```bash
python train_ST_D3QN.py --max_episodes 1000
```

**Training Process Details:**

- The UAV agent interacts with the environment, collects experiences, and learns optimal policies.
- The training script saves model checkpoints every 100 episodes.
- Training data and logs are saved in the specified directories.

### Testing the Model

After training, you can test the model using the `test_ST_D3QN.py` script:

```bash
python test_ST_D3QN.py
```

**Testing Script Details:**

- Loads the trained model from the checkpoints.
- Runs the UAV in the environment without exploration (greedy policy).
- Visualizes the UAV's path and interactions with the environment.
- Prints the total reward for each test episode and the average reward over all episodes.

## Results

- **Convergence**: The ST-D3QN algorithm demonstrates faster convergence compared to baseline algorithms.
- **Path Optimization**: Efficient path planning that minimizes collisions and path length.
- **Reward Optimization**: Improved handling of sparse rewards leading to better training efficiency.

*(For detailed results and plots, refer to the associated research article or generate them by running the training and testing scripts.)*

## Frequently Asked Questions

**1. Can I modify the environment parameters (e.g., grid size, obstacle positions)?**

Yes, you can modify the environment parameters in the `Env_ST_D3QN.py` file. Adjust variables such as `grid_size`, `self.obstacles`, and `self.dynamic_obstacles` as needed.

**2. How do I adjust the training hyperparameters?**

Training hyperparameters can be adjusted when initializing the `ST_D3QN` agent in the `train_ST_D3QN.py` script. Parameters such as `alpha` (learning rate), `gamma` (discount factor), and `epsilon` (exploration rate) can be modified.

**3. What if I encounter issues with the visualization window not displaying correctly?**

Ensure that `matplotlib` is installed correctly and that your environment supports GUI operations. If running on a remote server, you may need to configure X11 forwarding or run the scripts in a local environment.

**4. Can I use this code as a basis for extending to 3D path planning or multiple UAVs?**

The current implementation focuses on 2D path planning for a single UAV. However, the code structure is modular and can be extended to 3D environments or multi-UAV scenarios with additional modifications.

## Acknowledgments

This project is based on the research work conducted by Anping Yang, eta.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
