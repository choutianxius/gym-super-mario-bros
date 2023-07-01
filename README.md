# DDQN Agent for Super Mario Bros

Train a double deep Q neural network (DDQN) agent to play Nintento Super Mario Bros (simulated).

Based on [OpenAI Gym](https://www.gymlibrary.dev/).

## Requirements

- Python==3.9.13 (Python 3.9.x should be fine)
- gym==0.25.2 (It is not the latest version to keep compatible with gym-super-mario-bros)
- gym-super-mario-bros==7.4.0
- Pytorch needs manual configuration. See its [website](https://pytorch.org/get-started/locally/)

## Let the Agent Play!

### Trained Agent

~~~shell
python play.py
~~~

- [Example video](video/trained.mp4)

https://user-images.githubusercontent.com/100419654/222812866-7629a5ee-033f-4dc2-8ace-b5de884a9154.mp4

### Random Agent

- Pick a random action to proceed

~~~shell
python play_untrained.py
~~~

- [Example video](video/untrained.mp4)

https://user-images.githubusercontent.com/100419654/222812980-0f4e194a-9ba1-4a68-a0c3-5d0d78495e4d.mp4

### Metrics

- Average Reward During Training

  <img src="readme_images/reward_plot.jpg" alt="reward" style="zoom: 50%;" />

- Average Loss During Training

  <img src="readme_images/loss_plot.jpg" alt="loss" style="zoom: 50%;" />

- Expected Q Value During Training

  <img src="readme_images/q_plot.jpg" alt="loss" style="zoom: 50%;" />

- Average Duration of an Episode During Training

  <img src="readme_images/length_plot.jpg" alt="length" style="zoom:50%;" />

- Clear Times in 100 Episodes

  <img src="readme_images/clear_times.png" alt="length" style="zoom:25%;" />

- Mean Score in 100 Episodes

  <img src="readme_images/mean_scores.png" alt="length" style="zoom:25%;" />

## Training Details

### Policy Network Structure

Following the [2015 DeepMind DQN paper](https://doi.org/10.1038/nature14236)

1. Input: 4 × 84 × 84 float (transformed rgb representation of the screen: grayscale, resize and stack every 4 frames)
2. (conv2d + relu) * 3
3. flatten
4. (dense + relu) * 2
5. output: x float (x is the dimension of the action-space)

<img src="readme_images/convnet_fig.png" alt="length" />

### Algorithm: Double Deep Q Learning

$$
s = state, ~ s' = next ~ state\\
a = action, ~ a' = next ~ action\\
r = reward\\
\gamma = discount ~ factor\\
input = (s, a)\\
a' = argmax_{a} (Q_{online}(s', a))\\
target = r + \gamma \times Q_{target}(s', a')
$$

- Train the model with *input* and *target*
- Optimizer is Adam optimizer
- Loss function is Huber loss (`SmoothL1Loss`)
- Sample from a **replay memory** to get **mini-batches**
- Synchronize $Q_{target}$ with $Q_{online}$ every $C$ steps
- **$\epsilon$-greedy** with a exponentially decaying $\epsilon$ (during training)

#### Training Details

- $\gamma$ = 0.9
- $\epsilon$ exponentially degrades from 1 to 0.02, with degradation factor being 0.99995
- Trained for 50000 episodes
- Trained with 1 Nvidia RTX 4090 graphic card with 24 GB VRAM for about 48 hours
