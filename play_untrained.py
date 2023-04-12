import environment
from gym.wrappers.monitoring.video_recorder import VideoRecorder


record = True
mode = 'single_rgb_array' if record else 'human'
stage = 2

env = environment.base_env(mode, stage)

if record:
    recorder = VideoRecorder(
        env,
        path=f'./video/pre_untrained_{stage}.mp4',
    )
    recorder.metadata = env.metadata

env.reset()
state, reward, done, trunc, info = env.step(action=0)
episode_num = 0
step_count = 0
max_steps = 1000
while episode_num < 30:
    if done or trunc or (step_count >= max_steps):
        episode_num += 1
        step_count = 0
        state = env.reset()
    state, reward, done, trunc, info = env.step(env.action_space.sample())
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
