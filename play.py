from agent import Mario, MarioNet
import torch
from environment import wrapped
from gym.wrappers.monitoring.video_recorder import VideoRecorder


record = False
mode = 'single_rgb_array' if record else 'human'

env = wrapped(mode)

if record:
    recorder = VideoRecorder(
        env,
        path='./video/trained.mp4',
    )
    recorder.metadata = env.metadata

# Load model
save_path = 'save/2023-03-03T22-31-48/mario_net_1.chkpt'
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
while episode_num < 20:
    if done or trunc:
        episode_num += 1
        state = env.reset()
    action = mario.act(state)
    state, reward, done, trunc, info = env.step(action)
    if not record:
        env.render()
    else:
        recorder.capture_frame()

print('End')
if record:
    recorder.close()
else:
    env.close()
