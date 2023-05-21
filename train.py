import torch
from pathlib import Path
import datetime

from metric_logger import MetricLogger
from agent import Mario
from environment import wrapped


env = wrapped('rgb_array')

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)

# Train
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("save") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Mario(
    action_dim=env.action_space.n,
    save_dir=save_dir,
    enable_explore=True,
    train=True,
    memory_len=100000,
)
logger = MetricLogger(save_dir)

episodes = 50000
try:
    for e in range(episodes):
        state = env.reset()

        # play the game
        while True:
            # run the agent on the state
            action = mario.act(state)
            # agent performs action
            next_state, reward, done, trunc, info = env.step(action)
            # remember
            mario.cache(state, next_state, action, reward, done)
            # learn
            q, loss = mario.learn()
            # logging
            logger.log_step(reward, loss, q)
            # update state
            state = next_state
            # check if the game ends
            if done or info["flag_get"]:
                break

        logger.log_episode()

        if e % 20 == 0:
            logger.record(
                episode=e,
                epsilon=mario.exploration_rate,
                step=mario.curr_step
            )
except KeyboardInterrupt:
    pass
finally:
    # Save the final model when training ends
    mario.save()
    env.close()
