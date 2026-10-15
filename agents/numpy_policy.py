import numpy as np

# Policies trained on the 74-dim observation (before the peek/duel block): that layout is the
# current one minus features 73..93, so it can be sliced out of a 95-dim observation.
LEGACY_DIM, CURRENT_DIM = 74, 95


def adapt_obs(obs, obs_dim):
    if obs.size == CURRENT_DIM and obs_dim == LEGACY_DIM:
        return np.concatenate([obs[:73], obs[-1:]])
    return obs


class NumpyPolicy:
    """Torch-free inference for an SB3 PPO MlpPolicy (tanh MLP → MultiDiscrete logits).

    Loads the .npz written by tools/export_policies.py and exposes SB3's predict() signature.
    """

    def __init__(self, path):
        d = np.load(path)
        n = int(d["n_layers"])
        self.layers = [(d[f"w{i}"], d[f"b{i}"]) for i in range(n)]
        self.wa, self.ba = d["wa"], d["ba"]
        self.nvec = [int(v) for v in d["nvec"]]
        self.obs_dim = self.layers[0][0].shape[1]
        self.rng = np.random.default_rng()

    def predict(self, obs, deterministic=False):
        h = adapt_obs(np.asarray(obs, dtype=np.float32).reshape(-1), self.obs_dim)
        for w, b in self.layers:
            h = np.tanh(w @ h + b)
        logits = self.wa @ h + self.ba
        action, i = [], 0
        for n in self.nvec:
            l = logits[i:i + n]
            i += n
            if deterministic:
                action.append(int(np.argmax(l)))
            else:
                p = np.exp(l - l.max())
                action.append(int(self.rng.choice(n, p=p / p.sum())))
        return np.array(action, dtype=np.int64), None
