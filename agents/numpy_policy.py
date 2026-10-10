import numpy as np


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
        h = np.asarray(obs, dtype=np.float32).reshape(-1)
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
