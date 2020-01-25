# Discriminator Soft Actor Critic without Extrinsic Rewards
[Paper](https://arxiv.org/abs/2001.06808)

- DSAC Implementation using [ChainerRL](https://github.com/chainer/chainerrl).
- Our [Soft Actor Critic](https://arxiv.org/abs/1812.05905) is based on [ChainerRL implementation](https://github.com/chainer/chainerrl/blob/master/chainerrl/agents/soft_actor_critic.py).



## TODO

- [x] continuous action space 
- [ ] discrete action space



## Install

* `pip install -r requirements.txt`



## Usage

* Training [Soft Q Imitation Learning](https://arxiv.org/abs/1905.11108) (SQIL) and DSAC

  `python train_sqil.py [options]`
  
  * `--load-demo [dirname]` : replay buffer of demonstrations
  * `--absorb` : with absorbing state wrapper
  * `--reward_func` : use not constant rewards but generated rewards by a reward function.
  
  e.g.)  DSAC with absorbing state wrapper in AntBulletEnv-v0 (random seed = 1)
  
  * `python train_sqil.py --env AntBulletEnv-v0 --load-demo demos/4_episode/AntBulletEnv-v0 --absorb --reward-func --seed 1  `
  
  

## Requirement

python >= 3.7 and please see [requirements.txt](requirements.txt)

If you'd like to use GPU, please `pip install cupy-cudaOO`

​	In relation to your version of cuda `OO`, please see [the webpage of cupy](https://cupy.chainer.org/).
