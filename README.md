# BattleAI: 2D Arena with Trainable Agents

This project simulates a 2D environment where two robot agents fight each other. Agents can be manually controlled or trained using reinforcement learning (DQN). The simulation is rendered using Pygame, and the neural network is implemented in PyTorch.

## Features

- Manual or AI-based control of agents
- 2D arena with walls and collision detection
- Reward system for reinforcement learning
- Training with Deep Q-Network (DQN)
- Logging and visualization of training metrics
- Multiprocessing support for parallel training

## Project Structure

```
├── agent.py         # DQN agent and replay buffer
├── robot.py         # Robot logic and physics
├── network.py       # Neural network architecture
├── utils.py         # Helper functions: UI, state, hit checks
├── manual_main.py   # Manual control for testing vs AI
├── ver 0.py         # Main simulation with learning loop
├── train_main.py    # Multiprocessing training
├── main.py          # Main menu and mode selector
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/BattleAI.git
cd BattleAI
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

- **Manual control mode:**

```bash
python manual_main.py
```

- **Training mode (single arena):**

```bash
python ver\ 0.py
```

- **Multiprocessing training mode:**

```bash
python train_main.py
```

- **Main menu UI:**

```bash
python main.py
```

## Training

Agents are trained using a Deep Q-Network. Training sessions log scores, rewards, and model loss. Trained models are saved automatically for reuse.

## Technologies Used

- Python 3
- PyTorch
- Pygame
- Multiprocessing
- Matplotlib, Pandas

## Logging and Models

- All training data and models are stored in `runs/` or `logs/` folders
- Each run is saved in a separate subdirectory with a unique identifier

## Future Improvements

- Add support for PPO or A3C
- Enhanced visualizations
- Interactive statistics dashboard
- Network multiplayer mode

## License

MIT License

---

Developed by [Max](https://github.com/yourusername)

