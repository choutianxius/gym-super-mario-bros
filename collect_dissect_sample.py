"""
Use this script to generate
the dataset used for downstream dissect experiments.
"""

from agent import Mario, MarioNet
import torch
from environment import wrapped
from PIL import Image
from tqdm import tqdm


record = False
stage = 1

mode = 'rgb_array'
env = wrapped(mode, stage)

# Load model
# change to your model weights
save_path = f'save/2023-05-21T07-54-29/mario_net_{86}.chkpt'
data = torch.load(save_path)
state_dict = data.get('online_model')
output_dim = data.get('output_dim')

model = MarioNet(output_dim=output_dim)
model.load_state_dict(state_dict)
model.eval()
mario = Mario(action_dim=output_dim, model=model, enable_explore=True)

# Play
env.reset()
state, reward, done, trunc, info = env.step(action=0)

num_screenshots = 10000
steps_per_screenshot = 10
max_steps = num_screenshots * steps_per_screenshot


for _ in tqdm(range(max_steps)):
    if done or trunc:
        state = env.reset()
    action = mario.act(state)
    state, reward, done, trunc, info = env.step(action)
    if _ % steps_per_screenshot == 0:
        Image.fromarray(
            env.render(mode=mode)
        ).save(f'dissect_images/{_ // steps_per_screenshot}.png')

print('End')
env.close()
