# DDPG + HER

## Requirment

* python
* keras
* Mujoco-py
* NumPy
* gym

## Test Envirment

* Ubuntu 2004 LST
* python 3.8
* neural network update time 200 * 50 * 50
* initial state is limited with distance close to 0.05 (just for efficiency)
* openai environment *FetchPickAndPlace-v1*

## How to use

### How to train

* change the configuration variable *PLAY_MODEL* come from main.py to *False*
* Run main.py

### How to play

* Change the configuration variable *PLAY_MODEL* come from main.py to *True*
* Run main.py


## Model Explain

Combination with *Deep Deterministic Policy Gradient*(DDPG) and *Hindsight Experience Replay*(HER) algorithm.