from abc import ABC, abstractmethod

class BaseReward(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def on_hit(self, agent_idx, target_idx):
        pass

    @abstractmethod
    def on_hit_recv(self, agent_idx):
        pass

    @abstractmethod
    def on_kill(self, agent_idx, target_idx):
        pass

    @abstractmethod
    def on_death(self, agent_idx):
        pass

    @abstractmethod
    def on_timeout(self):
        pass

    @abstractmethod
    def per_tick(self, agent_idx, player, opponent):
        pass

    @abstractmethod
    def get_rewards(self):
        pass
