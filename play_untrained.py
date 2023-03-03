import environment
from gym.wrappers.monitoring.video_recorder import VideoRecorder


record = False
mode = 'single_rgb_array' if record else 'human'

env = environment.base_env(mode)

if record:
    recorder = VideoRecorder(
        env,
        path='./video/untrained.mp4',
    )
    recorder.metadata = env.metadata

env.reset()
state, reward, done, trunc, info = env.step(action=0)
for step in range(1000):
    if done or trunc:
        state = env.reset()
    state, reward, done, trunc, info = env.step(env.action_space.sample())
    if not record:
        env.render()
    else:
        recorder.capture_frame()

print('End')
if record:
    recorder.close()
else:
    env.close()
