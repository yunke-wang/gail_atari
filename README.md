# Atari-GAIL

This repository contains the PyTorch code for Generative Adversarial Imitation Learning (GAIL) with visual inputs, i.e. Atari games and visual dm-control.

## Requirements
Experiments were run with Python 3.6 and these packages:
* torch == 1.10.2
* gym == 0.19.0
* atari-py == 0.2.9

## Collect Expert Demonstrations

 * Train an Expert Policy with PPO
 ```
  python train_ppo.py --env-name "PongNoFrameskip-v4" --algo ppo --use-gae --lr 2.5e-4  --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01
 ```

 * Collect Expert Demonstrations
 ```
  python collect.py --env-name "PongNoFrameskip-v4"
 ```

We provide collected expert demonstrations in the following link. `Level 2' demonstrations are optimal demonstrations and `Level 1' demonstrations are sub-optimal demonstrations. [[Google Drive]](https://drive.google.com/drive/folders/1nlUf471Cp0g7N3JRy0lKnBJNnzJaEqY-?usp=sharing)

## Train GAIL

* Train GAIL with optimal demonstrations (without BC pre-training)
 ```
  python gail.py --gail --env-name "PongNoFrameskip-v4" --name pong --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01
 ```

* Train GAIL with optimal demonstrations (with BC pre-training)
 ```
  python gail.py --bc --gail --env-name "PongNoFrameskip-v4" --name pong --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01
 ```

* Train GAIL with imperfect demonstrations
 ```
  python gail.py --imperfect --bc --gail --env-name "PongNoFrameskip-v4" --name pong --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01
 ```

## Results

We train GAIL with 2000 optimal demonstrations. The results are as follow. 

| Method    | Pong | Seaquest | BeamRider | Hero | Qbert |
| :---      |:---: | :---:    |  :---:    | :---:| :---: |
|   BC      | -20.7(0.46) | 200.0(83.43) | 1028.4(396.37) | 7782.5(50.56) | 11420.0(3420.0) |
| GAIL      |  -1.73(18.1)| 1474.0(201.6)| 1087.6(559.09) | 13942.5(67.13)| 8027.27(24.9)   |
| GAIL+BC   | 21.0(0.0) | 1662.0(161.85) | 2306.4(1527.23) | 20020(22.91) | 13225.0(1347.22) |
| PPO(Best) | 21.0(0.0)| 1840(0.0)| 2637.45(1378.23)| 27814.09(46.01) | 15268.18(127.07) |

In our experiments, we find that using BC as a pre-training step can significantly improve the performance of GAIL in some Atari games.

## Acknowledegement
Our code structure is largely based on [Kostrikov's](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) implementation.
