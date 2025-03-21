import utils
from collections import deque
import numpy as np

import numpy as np
from abc import ABC, abstractmethod


import numpy as np


class Episode:
    def __init__(self) -> None:
        self.goals = []
        self.probs = []
        self.masks = []
        self.values = []
        self.states = []
        self.rewards = []
        self.actions = []

    def add(
        self,
        state: np.ndarray,
        reward: float,
        action,
        goal: bool,
        prob: float = None,
        value: float = None,
        masks: np.ndarray = None,
    ):
        self.goals.append(goal)
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

        if prob is not None:
            self.probs.append(prob)
        if value is not None:
            self.values.append(value)
        if masks is not None:
            self.masks.append(masks)

    def calc_advantage(self, gamma: float, gae_lambda: float) -> np.ndarray:
        n = len(self.rewards)
        advantages = np.zeros(n)
        for t in range(n - 1):
            discount = 1
            for k in range(t, n - 1):
                advantages[t] += (
                    discount
                    * (
                        self.rewards[k]
                        + gamma * self.values[k + 1] * (1 - int(self.goals[k]))
                    )
                    - self.values[k]
                )
                discount *= gamma * gae_lambda
        return list(advantages)

    def __len__(self):
        return len(self.goals)

    def total_reward(self) -> float:
        return sum(self.rewards)

class Buffer(ABC):
    def __init__(self, max_size: int, batch_size: int, shuffle: bool = True) -> None:
        super().__init__()
        self.shuffle = shuffle
        self.max_size = max_size
        self.batch_size = batch_size

    @abstractmethod
    def add(self, *args) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def get_len(self) -> int:
        pass

    @abstractmethod
    def sample(self):
        pass

    def __len__(self):
        return self.get_len()
class BufferPPO(Buffer):
    def __init__(
        self,
        max_size: int,
        batch_size: int,
        gamma: float,
        gae_lambda: float,
        shuffle: bool = True,
    ) -> None:
        super().__init__(max_size, batch_size, shuffle)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.episodes = deque(maxlen=max_size)
        self.advantages = deque(maxlen=max_size)

    def add(self, episode: Episode):
        self.episodes.append(episode)
        self.advantages.append(episode.calc_advantage(self.gamma, self.gae_lambda))

    def clear(self) -> None:
        self.episodes.clear()
        self.advantages.clear()

    def get_len(self) -> int:
        return len(self.episodes)

    def sample(self):
        probs = sum(map(lambda x: x.probs, self.episodes), [])
        goals = sum(map(lambda x: x.goals, self.episodes), [])
        masks = sum(map(lambda x: x.masks, self.episodes), [])
        values = sum(map(lambda x: x.values, self.episodes), [])
        states = sum(map(lambda x: x.states, self.episodes), [])
        actions = sum(map(lambda x: x.actions, self.episodes), [])
        rewards = sum(map(lambda x: x.rewards, self.episodes), [])
        advantages = sum(self.advantages, [])

        batches = utils.make_batch_ids(
            n=len(states), batch_size=self.batch_size, shuffle=self.shuffle
        )

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(goals),
            np.array(probs),
            np.array(values),
            np.array(masks),
            np.array(advantages),
            batches,
        )