import torch, random, copy
import numpy as np
from collections import deque
from model import MarioNet


# Agent. Able to act, remember and learn.
class Mario:
    """Mario agent."""
    def __init__(
            self,
            action_dim,
            save_dir="",
            memory_len=10000,
            batch_size=32,
            exploration_rate_decay=0.999975,
            exploration_rate_min=0.1,
            enable_explore=False,
            model=None,
            train=False,
        ):
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.memory = deque(maxlen=memory_len)
        self.batch_size = batch_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.online_model = MarioNet(self.action_dim).float()\
                            if model is None\
                            else copy.deepcopy(model)
        self.online_model = self.online_model.to(device=self.device)

        self.train = train
        self.target_model = copy.deepcopy(self.online_model)\
                            if self.train\
                            else None
        if self.target_model:
            for p in self.target_model.parameters():
                p.requires_grad = False

        # When training, enable enable_explore; otherwise disable
        self.enable_explore = enable_explore
        self.exploration_rate = 1
        self.exploration_rate_decay = exploration_rate_decay
        self.exploration_rate_min = exploration_rate_min
        self.curr_step = 0

        # Save MarioNet every # experiences gained (steps)
        self.save_every = memory_len * 10

        self.gamma = .9
        self.optimizer = torch.optim.Adam(
            self.online_model.parameters(),
            lr=0.00025
        )
        self.loss_fn = torch.nn.SmoothL1Loss()

        # min. experiences before training
        self.burnin = memory_len // 10
        # no. of experiences between updates to Q_online
        self.learn_every = 3
        # no. of experiences between Q_target & Q_online sync
        self.sycn_every = self.learn_every * 50

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action.
        With a probability of epsilon, explore (random action);
        With a probability of (1 - epsilon), exploit (optimal action)
        """
        epsilon = self.exploration_rate\
                  if self.train\
                  else self.exploration_rate_min
        # explore
        if self.enable_explore and (np.random.rand() < epsilon):
            action_idx = np.random.randint(self.action_dim)

        # exploit
        else:
            # Calculate Q(s, a) for every a
            # return argmax_a(Q(s, a))
            state = state[0].__array__()\
                    if isinstance(state, tuple)\
                    else state.__array__()
            state = torch.tensor(
                state,
                device=self.device
            ).unsqueeze(0)
            action_values = self.online_model(state)
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(
            self.exploration_rate,
            self.exploration_rate_min
        )

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        self.memory.append(
            (state, next_state, action, reward, done, )
        )

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(
            torch.stack,
            zip(*batch)
        )
        return (
            state,
            next_state,
            action.squeeze(),
            reward.squeeze(),
            done.squeeze()
        )

    def td_estimate(self, state, action):
        """Calculate input Q value during training"""
        current_Q = self.online_model(state)[
            np.arange(0, self.batch_size),
            action
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """Calcuate target Q value during training"""
        # a' = argmax_a(Q_online(s', a))
        # target = reward + gamma * Q_target(s', a') if not terminal
        # target = reward otherwise
        next_state_Q = self.online_model(next_state)
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.target_model(next_state)[
            np.arange(0, self.batch_size),
            best_action
        ]
        return (
            reward +
            (1 - done.float()) * self.gamma * next_Q
        ).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.target_model.load_state_dict(
            self.online_model.state_dict()
        )

    def save(self):
        """Save model params and hyperparams to reuse"""
        save_path = (
            self.save_dir / "mario_net_"\
                f"{int(self.curr_step // self.save_every)}"\
                ".chkpt"
        )
        torch.save(
            dict(
                online_model=self.online_model.state_dict(),
                output_dim=self.action_dim,
                exploration_rate=self.exploration_rate,
            ),
            save_path
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        """
        Update online action-value (Q) function with a batch of experiences.
        Use double DQN algorithm.
        """
        if self.curr_step % self.sycn_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Calculate TD estimate
        td_est = self.td_estimate(state, action)

        # Calculate TD target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)
