# Overview
This project contains three Python files, baseline.py, sarsa.py and DDQN.py. The baseline.py file contains a baseline algorithm that uses random actions as the policy to play the "Crazy Climber" game in OpenAI Gym. The DDQN.py and sarsa.py files contain an implementation of the Double Deep Q-Network and Sarsa algorithm respectively.

# Dependencies

To run this project, you need Python 3.6 or later and the following Python packages:

* numpy
* gymnasium
* tensorflow
* matplotlib

# Installation
1. Install Python 3.6 or later.
2. Install the necessary Python packages:

`pip install numpy gym tensorflow matplotlib`

# Usage
To run the baseline algorithm, run the following command:

`python baseline.py`

To run the DDQN algorithm, run the following command:

`python DDQN.py`

To run the Sarsa algorithm, run the following command:

`python sarsa.py`

# Results

The baseline.py script generates a plot of the rewards obtained by the algorithm during each episode of the game, which is saved as 'baseline.png' under the folder 'Results'. The rewards obtained in each episode are also appended to the baseline_results.txt file under the folder 'Text files'.

The DDQN.py script generates a plot of the rewards obtained by the algorithm during each episode of the game, which is saved as DDQN.png under the folder 'Results'. The rewards obtained in each episode are also appended to the DDQN_results.txt file. The script also saves the video recordings of the game played by the agent in the ./recordings directory.

The sarsa.py script generates a plot of the rewards obtained by the algorithm during each episode of the game, which is saved as sarsa.png under the folder 'Results'. The rewards obtained in each episode are also appended to the sarsa_results.txt file. The script also saves the video recordings of the game played by the agent in the ./recordings directory.

Note that the DDQN.py and sarsa.py scripts may take several hours to complete, depending on your hardware.
