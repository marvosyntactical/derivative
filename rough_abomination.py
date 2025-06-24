# Rough-Heston-Hawkes-SLV-Impact monster – JAX implementation
# --------------------------------------------------------------
# Author: ChatGPT o3 & Marvin Koß – 2025-06-24
# Licence: MIT.

"""
MODULAR DESIGN (v2 – now with training loop)
===========================================
This file now covers three pillars:
    1.  Model core (unchanged): kernel mixture → affine Riccati → FFT pricer;
        Hawkes–SLV–impact Monte‑Carlo for path generation.
    2.  **Calibration loop**: global CMA‑ES warm‑start + Adam fine‑tune.
    3.  **RL hedging boilerplate**: minimal actor–critic training loop
        (TD‑error, Optax optimiser) fed by on‑the‑fly Monte‑Carlo paths.

GPU / TPU acceleration is automatic once JAX is installed with CUDA.

Quick start (Linux + CUDA):
    $ pip install "jax[cuda11_local]" diffrax optax rich cma gymnasium
    $ python rough_abomination.py   # self‑test + tiny RL session

Real data pipeline outline is at the bottom of the file (# 9).
"""
################################################################################
# 1 ▸ Imports & global utilities (unchanged)
################################################################################
from __future__ import annotations
import jax, jax.numpy as jnp, jax.random as jrd
import jax.scipy as jsp
from jax import jit, vmap, grad
import optax
import diffrax as dfx
import functools, math, sys, time
from typing import NamedTuple, Callable
from tqdm.auto import tqdm

try:
    import cma  # optional global optimiser
except ImportError:
    cma = None
try:
    import gymnasium as gym  # for RL env wrapper
except ImportError:
    gym = None  # RL part will complain politely
################################################################################
# 2 ▸ Kernel approximation – exponential mixture (unchanged)
################################################################################
class KernelMix(NamedTuple):
    weights: jnp.ndarray  # shape (M,)
    lambdas: jnp.ndarray  # shape (M,)

@functools.partial(jit, static_argnums=(1,))
def fit_mixture(H: float, M: int = 10, T_max: float = 1.0, lam_ratio: float = 1e3) -> KernelMix:
    """Return (w_i, λ_i) s.t.  t↦t^{H-1/2} ≈ Σ w_i e^{-λ_i t} on [0,T_max]."""
    lam_min = 1.0 / T_max
    lam_max = lam_min * lam_ratio
    lam = jnp.geomspace(lam_min, lam_max, M)
    t = jnp.linspace(1e-6, T_max, 10 * M)
    K = t ** (H - 0.5)
    A = jnp.exp(-t[:, None] * lam[None, :])
    w = jnp.ones(M) / M
    lr = 1e-2
    for _ in range(2_000):
        grad_w = (A.T @ (A @ w - K)) / t.size
        w = jnp.clip(w - lr * grad_w, 0.0)
    w = w / jnp.sum(w) * jnp.sum(K) / jnp.sum(A @ w)
    return KernelMix(w, lam)
################################################################################
# 3 ▸ Riccati ODE (affine Volterra – after Markovianisation) (unchanged)
################################################################################
class RHParams(NamedTuple):
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float
    mix: KernelMix

@functools.partial(jit, static_argnums=(2,))
def riccati_rhs(t, y, args, *, backend="c64"):
    kappa, theta, xi, rho, w, lam = args
    M = w.size
    phi = y[0]
    psi = y[1 : 1 + M]
    psi_dot = -lam * psi + 0.5 * xi**2 * psi**2 - kappa * psi + kappa * theta
    x = (w * psi).sum()
    phi_dot = 0.5 * x ** 2
    return jnp.concatenate((jnp.array([phi_dot]), psi_dot))


def solve_riccati(u: complex, T: float, p: RHParams):
    w, lam = p.mix
    args = (p.kappa, p.theta, p.xi, p.rho, w, lam)
    y0 = jnp.concatenate((jnp.array([0.0 + 0.0j]), jnp.zeros_like(w)))
    term = dfx.diffeqsolve(
        dfx.ODETerm(lambda t, y, args: riccati_rhs(t, y, args)),
        solver=dfx.Dopri5(),
        t0=0.0,
        t1=T,
        dt0=1e-3,
        y0=y0,
        args=args,
    )
    phi_T = term.ys[0]
    psi_T = term.ys[1:]
    log_cf = 1j * u * 0.0 + phi_T + (psi_T * p.v0 * w).sum()
    return jnp.exp(log_cf)
################################################################################
# 4 ▸ Carr–Madan FFT pricer (unchanged)
################################################################################
@jit
def price_via_fft(params: RHParams, T: float, strikes: jnp.ndarray, S0: float, alpha: float = 1.5):
    N = 2 ** 12
    eta = 0.25
    u = jnp.arange(N) * eta
    cf = vmap(lambda uu: solve_riccati(uu - 1j * (alpha + 1), T, params))(u)
    integrand = cf * jnp.exp(-1j * u * jnp.log(strikes[0])) * eta
    fft = jnp.fft.fft(integrand).real
    k = jnp.log(strikes / S0)
    prices = jnp.exp(-alpha * k) / math.pi * fft[: strikes.size]
    return prices
################################################################################
# 5 ▸ Monte-Carlo generator with Hawkes jumps, SLV, impact (unchanged)
################################################################################
class SimConfig(NamedTuple):
    T: float
    steps: int
    mu: float
    r: float
    q: float
    gamma: float  # impact coef
    J_mean: float
    hawkes_eta: float  # baseline intensity
    hawkes_beta: float  # decay

@functools.partial(jit, static_argnums=(2,))
def simulate_paths(key, params: RHParams, cfg: SimConfig, paths: int = 4096):
    dt = cfg.T / cfg.steps
    w, lam = params.mix
    M = w.size
    keys = jrd.split(key, cfg.steps)
    S = jnp.full((paths,), 1.0)
    V = jnp.full((paths,), params.v0)
    Y = jnp.zeros((M, paths))
    Q = jnp.zeros(paths)
    lam_jump = jnp.full(paths, cfg.hawkes_eta)
    for t in range(cfg.steps):
        z_v = jrd.normal(keys[t], (paths,))
        z_s = jrd.normal(keys[t] + 1, (paths,))
        Y = Y + (-lam[:, None] * Y) * dt + params.xi * jnp.sqrt(jnp.clip(V, 1e-12)) * z_v * jnp.sqrt(dt)
        V = (w[:, None] * Y).sum(0)
        jump_prob = jnp.clip(lam_jump * dt, 0.0, 0.5)
        jumps = jrd.bernoulli(keys[t] + 2, jump_prob)
        S = S * jnp.exp((cfg.mu - 0.5 * V) * dt + jnp.sqrt(jnp.clip(V, 1e-12)) * z_s * jnp.sqrt(dt)) * jnp.exp(cfg.J_mean * jumps)
        lam_jump = cfg.hawkes_eta + lam_jump * jnp.exp(-cfg.hawkes_beta * dt) + jumps * cfg.hawkes_beta
        Q = Q - S  # naive delta hedge
        S = S * jnp.exp(-cfg.gamma * Q * dt)
    return S, V
################################################################################
# 6 ▸ Calibration objective and optimisation (unchanged)
################################################################################
class CalibSettings(NamedTuple):
    strikes: jnp.ndarray
    maturities: jnp.ndarray
    market_prices: jnp.ndarray  # shape (T, K)
    S0: float

@jit
def objective(params_vec, settings: CalibSettings, mix: KernelMix):
    p = RHParams(*params_vec, mix=mix)
    pred_prices = vmap(lambda T:
                       price_via_fft(p, T, settings.strikes, settings.S0))(settings.maturities)
    return ((pred_prices - settings.market_prices) ** 2).mean()


def calibrate_jax(settings: CalibSettings, mix: KernelMix, seed=0,
                  global_cma=True, max_iter=3_000, log_every=100):
    lb = jnp.array([1e-3, 1e-3, 1e-3, -0.999, 1e-4]); ub = jnp.array([15, 5, 5, 0.999, 2])
    # —— CMA warm‑start ——
    if global_cma and cma is not None:
        es = cma.CMAEvolutionStrategy((lb + ub) / 2, 0.3*(ub-lb).mean(), {'bounds':[lb.tolist(), ub.tolist()]})
        for _ in range(8):
            cand = es.ask()
            losses = [float(_loss(jnp.array(x), settings, mix)) for x in cand]
            es.tell(cand, losses)
        params = jnp.array(es.best.xbounds)
    else:
        params = (lb + ub) / 2
    # —— Adam fine‑tune ——
    opt = optax.adam(5e-3)
    opt_state = opt.init(params)
    val_and_grad = jax.value_and_grad(_loss)
    iterator = tqdm(range(max_iter), desc="Adam", disable=(tqdm is None)) if tqdm else range(max_iter)
    for it in iterator:
        loss, g = val_and_grad(params, settings, mix)
        updates, opt_state = opt.update(g, opt_state)
        params = jax.tree_util.tree_map(lambda p, u: jnp.clip(p - u, lb, ub), params, updates)
        if tqdm:
            iterator.set_postfix({"loss": float(loss)})
        elif it % log_every == 0:
            print(f"iter {it:5d} | loss {float(loss):.6e}")
    return params


################################################################################
# 7 ▸ RL hedging boilerplate (new)
################################################################################
if gym is not None:
    class RoughHedgeEnv(gym.Env):
        """A toy environment: hedge 1‑share option with delta under impact."""
        metadata = {"render_modes": []}
        def __init__(self, params: RHParams, cfg: SimConfig, option_T: float, strikes: jnp.ndarray):
            super().__init__()
            self.params, self.cfg = params, cfg
            self.action_space = gym.spaces.Box(-10.0, 10.0, (1,), dtype=float)
            self.observation_space = gym.spaces.Box(-jnp.inf, jnp.inf, (3,), dtype=float)
            self.key = jrd.PRNGKey(42)
            self.S, self.V = None, None
            self.t = 0
            self.steps = cfg.steps
            self.dt = cfg.T / cfg.steps
        def reset(self, seed=None, options=None):
            self.key, sub = jrd.split(self.key)
            self.S, self.V = simulate_paths(sub, self.params, self.cfg, paths=1)
            self.S = float(self.S[0]); self.V = float(self.V[0])
            self.t = 0
            return jnp.array([self.S, self.V, self.t*self.dt]), {}
        def step(self, action):
            hedge = jnp.clip(action[0], -10.0, 10.0)
            pnl = -0.5 * self.cfg.gamma * hedge**2 * self.dt  # impact cost
            self.t += 1
            done = self.t >= self.steps
            obs = jnp.array([self.S, self.V, self.t*self.dt])
            return obs, float(pnl), done, False, {}

    def actor(params, obs):
        return jnp.tanh(jnp.dot(obs, params[:3]) + params[3]) * 10.0

    def critic(params, obs):
        return jnp.dot(obs, params[:3]) + params[3]

    def rl_train(params_env: RHParams, cfg: SimConfig, epochs: int = 100):
        env = RoughHedgeEnv(params_env, cfg, option_T=cfg.T, strikes=jnp.array([1.0]))
        key = jrd.PRNGKey(0)
        actor_p = jrd.normal(key, (4,)) * 0.1
        critic_p = jrd.normal(key+1, (4,)) * 0.1
        opt_actor = optax.adam(1e-2)
        opt_critic = optax.adam(1e-2)
        sa, sc = opt_actor.init(actor_p), opt_critic.init(critic_p)
        gamma = 0.995
        for ep in range(epochs):
            obs, _ = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                act = actor(actor_p, obs)
                obs2, reward, done, _, _ = env.step(jnp.array([act]))
                td_target = reward + gamma * critic(critic_p, obs2) * (1.0 - done)
                td_error = td_target - critic(critic_p, obs)
                # critic update
                grads_c = grad(lambda p: (td_error**2))(critic_p)
                critic_p, sc = opt_critic.update_and_get_new_params(grads_c, sc, critic_p)
                # actor update
                grads_a = grad(lambda p: -critic(critic_p, obs))(actor_p)
                actor_p, sa = opt_actor.update_and_get_new_params(grads_a, sa, actor_p)
                obs = obs2
                ep_reward += reward
            if ep % 10 == 0:
                print(f"Episode {ep}: reward {ep_reward:6.4f}")

################################################################################
# 8 ▸ Real‑data helper – download IV surface (yfinance)
################################################################################
# yfinance import can fail for many binary‑compat reasons (numpy ↔ pandas).
try:
    import yfinance as yf  # noqa: F401
except Exception as e:  # ImportError, ValueError from binary mismatch, etc.
    print(f"[warn]   yfinance unavailable → real‑data loader disabled ({e})")
    yf = None

def load_iv_surface_yf(ticker: str, expiry: str):
    """Return (strikes, maturities, prices) from Yahoo ‑ *toy quality* only."""
    if yf is None:
        raise ImportError("pip install yfinance")
    opt = yf.Ticker(ticker).option_chain(expiry)
    calls = opt.calls
    S0 = yf.Ticker(ticker).history(period="1d").Close.iloc[-1]
    strikes = jnp.array(calls.strike.values, dtype=float)
    iv = jnp.array(calls.impliedVolatility.values, dtype=float)
    T = (jnp.datetime64(expiry) - jnp.datetime64("today")) / jnp.timedelta64(365, "D")
    prices = 0.5 * (calls.bid.values + calls.ask.values)
    return strikes, jnp.array([T]), jnp.array([prices]), float(S0)


################################################################################
# 9 ▸ Demo / CLI – choose: calibrate or RL train on synthetic data
################################################################################
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["calib", "rl"], default="calib")
    args = parser.parse_args()
    # Rough kernel mix
    S0 = 100.0
    strikes = jnp.linspace(70, 130, 25)
    maturities = jnp.array([0.25, 0.5, 1.0])
    mix = fit_mixture(0.1, M=8, T_max=float(maturities.max()))
    true_p = RHParams(1.5, 0.04, 0.6, -0.7, 0.04, mix)
    if args.mode == "calib":
        mkt_px = vmap(lambda T: price_via_fft(true_p, T, strikes, S0))(maturities)
        settings = CalibSettings(strikes, maturities, mkt_px, S0)
        est_p = calibrate_jax(settings, mix, global_cma=False)
        print("Est Params:", est_p)
    else:
        cfg = SimConfig(T=1.0, steps=252, mu=0.0, r=0.0, q=0.0, gamma=5e-4, J_mean=-0.05,
                        hawkes_eta=0.3, hawkes_beta=4.0)
        rl_train(true_p, cfg, epochs=50)
