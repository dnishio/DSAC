# Discriminator Soft Actor Critic without External Rewards

- DSAC Implementation using [ChainerRL](https://github.com/chainer/chainerrl).
- Our Soft Actor Critic is based on [ChainerRL implementation](https://github.com/chainer/chainerrl/blob/master/chainerrl/agents/soft_actor_critic.py).



## TODO

- [x] continuous action space 
- [ ] discrete action space



## Install

* `pip install -r requirements.txt`



## Usage

* Training SQIL and DSAC

  `python train_sqil.py [options]`
  
  * `--load-demo [dirname]` : replay buffer of demonstrations
  * `--absorb` : with absorbing state wrapper
  * `--reward_func` : use not constant rewards but generated rewards by a reward function.
  
  e.g.)  DSAC with absorbing state wrapper in AntBulletEnv-v0 
  
  * `python train_sqil.py --env AntBulletEnv-v0 --load-demo demos/4_episode/AntBulletEnv-v0 --absorb --reward-func --seed 1  `
  
  

## Requirement

python >= 3.7 and see [requirements.txt](requirements.txt)

If you'd like to use GPU, please `pip install cupy-cudaOO`

​	see [the webpage of cupy](https://cupy.chainer.org/).