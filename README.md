# Ball-Sort-Learning-Environment

This project provides a Learning Environment for the Ball Sort Color Puzzle Game.

### Installation

To install the environment clone the repo and:

1) install the required libraries
```
pip install -r requirements.txt
```

2) install the environment package:
```
pip install -e ./YOUR_REPO_DIRECTORY
```

Then in your script import the **envs** directory so you can instantiate the environment like:
```
env = BallSortEnv()
```
or
```
env = gym.make("ballsort-v0")
```
or
```
env = make_vec_env("ballsort-v0", n_envs=4)
```
for building parallel environments (works very well for training agents from Stable Baselines 3)

### Examples

Run a **random agent**
```
python .\gym_ballsort\src\random_agent.py
```

Test a pre-trained PPO agent for solving the level 7 puzzle
```
python .\gym_ballsort\src\test_env.py
```

### Folders

**/levels** contains the json files that define different puzzles.

**/models** stores the saved agents

**/logs** stores the tensorboard logs that can be inspected with the following command (in local machine):
```
tensorboard --logdir=./logs/YOUR DIRECTORY HERE/ --host=127.0.0.1
```

**/gym_ballsort/envs**  contains the environment package. The environment definition is in: **/gym_ballsort/envs/ballsort.py**

**/src** contains the RL algorithms for training custom DQN, PPO models or other ones from Stable Baselines 3
    - **/src/main_sb3.py** contains the framework for training Stable Baselines agents
    



