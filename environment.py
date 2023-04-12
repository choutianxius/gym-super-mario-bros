import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation, FrameStack


def base_env(mode, stage = 1):
    # gym version compatibility
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make(f'SuperMarioBros-1-{stage}-v0', new_step_api=True)
    else:
        env = gym_super_mario_bros.make(
            f'SuperMarioBros-1-{stage}-v0',
            render_mode=mode,
            apply_api_compatibility=True,
            disable_env_checker=True
        )

    # Limit the action-space
    # See the documents
    # https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/actions.py
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env


def wrapped(mode, stage = 1):
    env = base_env(mode, stage)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)
    return env

