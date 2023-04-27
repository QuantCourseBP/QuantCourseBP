import numpy as np
from scipy.stats import norm

class Index:

    def __init__(self,
                 name: str,
                 level: float):
        self.name = name
        self.level = level

        def get_name() -> str:
            return self.name

        def get_level() -> float:
            return self.level


class FlatVol:
    def __init__(self,
                 index: Index,
                 vol: float,
                 drift: float):
        self.index = index
        self.vol = vol
        self.drift = drift


class MCEngineFlatVol:

    def __init__(self,
                 index: Index,
                 model: FlatVol,
                 seed: int,
                 num_of_paths: int,
                 tenors: list[float],
                 antithetic: bool,
                 standardize: bool):
        self.index = index
        self.seed = seed
        self.num_of_paths = num_of_paths
        self.model = model
        self.tenors = [0.0] + tenors
        self.num_of_tenors = len(tenors)
        self.antithetic = antithetic
        self.standardize = standardize

    def generate_std_norm(self) -> np.array:

        np.random.seed(self.seed)

        if self.antithetic:
            rnd1 = np.random.standard_normal(size=(int(self.num_of_paths / 2), self.num_of_tenors))
            rnd2 = -rnd1
            rnd = np.concatenate((rnd1, rnd2), axis=0)
            if self.num_of_paths % 2 == 1:
                zeros = np.zeros((1, self.num_of_tenors))
                rnd = np.concatenate((rnd, zeros), axis=0)

        else:
            rnd = np.random.standard_normal(size=(self.num_of_paths, self.num_of_tenors))

        if self.standardize:
            mean = np.mean(rnd)
            std = np.std(rnd)
            rnd = (rnd - mean) / std
        return rnd

    def create_spot_paths(self) -> np.array:

        std_norm = self.generate_std_norm()

        rf_rate = self.model.drift
        vol = self.model.vol
        log_proc_drift = rf_rate - 1 / 2 * vol * vol

        s0 = self.index.level

        log_ret_paths = np.zeros((self.num_of_paths, self.num_of_tenors + 1))

        for path in range(self.num_of_paths):
            log_ret_paths[path][0] = 0

        for path in range(self.num_of_paths):
            for tenor_idx in range(1, self.num_of_tenors + 1):
                d_time = self.tenors[tenor_idx] - self.tenors[tenor_idx - 1]
                log_ret_paths[path][tenor_idx] = log_ret_paths[path][
                                                     tenor_idx - 1] + log_proc_drift * d_time + vol * np.sqrt(d_time) * \
                                                 std_norm[path][tenor_idx - 1]
        return s0 * np.exp(log_ret_paths)


def call_payoff(s, k):
    return max(s - k, 0)


def discount_factor(r, t):
    return np.exp(-r * t)


def black_scholes(S, K, T, r, sigma, option='call'):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be either 'call' or 'put'.")


spx = Index('spx', 1000)
model = FlatVol(spx, 0.2, 0.05)

mcengine = MCEngineFlatVol(spx, model, int(np.random.rand()*10000), 1024, [1], True, True)

paths = mcengine.create_spot_paths()

payoff = np.array([call_payoff(spot, 1000) for spot in paths[:, -1]])
disc_payoff = payoff * discount_factor(model.drift, mcengine.tenors[-1])
pv = np.mean(disc_payoff)
print(pv)

bs_pv = black_scholes(spx.level, 1000, mcengine.tenors[-1], model.drift, model.vol)
print(bs_pv)