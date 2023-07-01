"""
Play with a trained agent
"""

from agent import Mario, MarioNet
import torch
from environment import wrapped
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--record",
                    help="If specified, record the playing process"
                    " into a video, instead of render it in real time",
                    action="store_true")
parser.add_argument("--stage",
                    help="Stage number",
                    type=int,
                    default=1)
args = parser.parse_args()
record = bool(args.record)
stage = args.stage

mode = 'single_rgb_array' if record else 'human'
env = wrapped(mode, stage=stage, world=1)

if record:
    recorder = VideoRecorder(
        env.unwrapped,
        path=f'./video/defense_stage{stage}_unwrapped.mp4',
    )
    recorder.metadata = env.metadata

# Load model
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

episode_num = 0
step_count = 0
max_steps = 1000
while episode_num < 10:
    if done or trunc or step_count >= max_steps:
        episode_num += 1
        state = env.reset()
        step_count = 0
    action = mario.act(state)
    state, reward, done, trunc, info = env.step(action)
    step_count += 1
    if not record:
        env.render()
    else:
        recorder.capture_frame()

print('End')
if record:
    recorder.close()
else:
    env.close()
