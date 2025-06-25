import jax.numpy as jnp
import yfinance as yf     # pip install yfinance
import matplotlib.pyplot as plt
from rough_abomination import (load_iv_surface_yf,      # already defined earlier
                               fit_mixture, CalibSettings,
                               calibrate_jax, price_fft, RHParams)

# ----- pull one expiry -----
ticker  = "SPY"
expiry  = "2025-09-20"       # yyyy-mm-dd shown on Yahoo’s option chain
strikes, maturities, mkt_px, S0 = load_iv_surface_yf(ticker, expiry)

# strikes  -> 1‑D array of K
# maturities -> length‑1 array with time‑to‑expiry T
# mkt_px   -> shape (1, K) array of mid‑market call prices

# build the rough‑kernel mixture for that expiry horizon
mix = fit_mixture(H=0.1, M=8, T_max=float(maturities.max()))

settings = CalibSettings(strikes=strikes,
                         maturities=maturities,
                         market_prices=mkt_px,
                         S0=S0)

params_vec = calibrate_jax(settings, mix,
                           global_cma=True,   # CMA warm‑start
                           max_iter=3000)     # Adam fine‑tune

# unpack into named tuple (kappa, theta, xi, rho, v0, mix)
p = RHParams(*params_vec, mix=mix)



# model prices for that maturity
fit_px = price_fft(p, maturities[0], strikes, S0)

fig, ax = plt.subplots()
ax.plot(strikes, mkt_px[0], "o", label="market mid")
ax.plot(strikes, fit_px,  "-", label="rough‑Heston fit")
ax.set_xlabel("Strike"); ax.set_ylabel("Call price")
ax.set_title(f"{ticker} {expiry} – model vs. market")
ax.legend(); ax.grid(True)
plt.show()
