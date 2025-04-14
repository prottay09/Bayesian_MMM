# %%
# Bayesian MMM with Adstock, Saturation, and Seasonality

import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
# %%
# -----------------------------
# 1. Simulate Data
# -----------------------------
n = 100
np.random.seed(42)
tv_spend = np.random.normal(100, 20, size=n)
search_spend = np.random.normal(50, 10, size=n)
noise = np.random.normal(0, 10, size=n)
sales = 5 + 0.3 * tv_spend + 0.5 * search_spend + noise

# Time + Seasonality
week = np.arange(n)
T = 52  # annual seasonality for weekly data
sin_season = np.sin(2 * np.pi * week / T)
cos_season = np.cos(2 * np.pi * week / T)

# DataFrame
df = pd.DataFrame({
    "tv_spend": tv_spend,
    "search_spend": search_spend,
    "sales": sales,
    "week": week,
    "sin_season": sin_season,
    "cos_season": cos_season
})
# %%
# -----------------------------
# 2. Define Hill/Saturation Function
# -----------------------------
def hill_transform(x, alpha, theta):
    return (x ** alpha) / (x ** alpha + theta ** alpha)

# -----------------------------
# 3. PyMC Model with Adstock + Saturation + Seasonality
# -----------------------------
with pm.Model() as mmm_sat:
    # Decay (adstock)
    decay_tv = pm.Beta("decay_tv", 2, 2)
    decay_search = pm.Beta("decay_search", 2, 2)

    def adstock(x, decay):
        result = pt.zeros_like(x)
        result = pt.set_subtensor(result[0], x[0])
        for t in range(1, x.shape[0]):
            result = pt.set_subtensor(result[t], x[t] + decay * result[t - 1])
        return result

    tv_adstock = adstock(df["tv_spend"].values, decay_tv)
    search_adstock = adstock(df["search_spend"].values, decay_search)

    # Saturation parameters
    alpha_tv = pm.HalfNormal("alpha_tv", sigma=1)
    theta_tv = pm.HalfNormal("theta_tv", sigma=10)
    alpha_search = pm.HalfNormal("alpha_search", sigma=1)
    theta_search = pm.HalfNormal("theta_search", sigma=10)

    tv_saturated = hill_transform(tv_adstock, alpha_tv, theta_tv)
    search_saturated = hill_transform(search_adstock, alpha_search, theta_search)

    # Coefficients
    intercept = pm.Normal("intercept", mu=0, sigma=5)
    beta_tv = pm.Normal("beta_tv", mu=0, sigma=1)
    beta_search = pm.Normal("beta_search", mu=0, sigma=1)
    beta_sin = pm.Normal("beta_sin", mu=0, sigma=1)
    beta_cos = pm.Normal("beta_cos", mu=0, sigma=1)

    # Noise
    sigma = pm.Exponential("sigma", 1)

    # Expected sales
    mu = (
        intercept +
        beta_tv * tv_saturated +
        beta_search * search_saturated +
        beta_sin * df["sin_season"] +
        beta_cos * df["cos_season"]
    )

    # Likelihood
    y_obs = pm.Normal("sales", mu=mu, sigma=sigma, observed=df["sales"])

    # Sampling
    trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)

# %%
# -----------------------------
# 4. Posterior Analysis
# -----------------------------
az.plot_posterior(trace, var_names=[
    "beta_tv", "beta_search", "decay_tv", "decay_search",
    "alpha_tv", "theta_tv", "alpha_search", "theta_search"
])
plt.show()

# -----------------------------
# 5. Plot Saturation Curve
# -----------------------------
alpha_tv_mean = trace.posterior["alpha_tv"].mean().item()
theta_tv_mean = trace.posterior["theta_tv"].mean().item()

x = np.linspace(0, df["tv_spend"].max(), 100)
y = (x ** alpha_tv_mean) / (x ** alpha_tv_mean + theta_tv_mean ** alpha_tv_mean)

plt.plot(x, y)
plt.title("TV Spend Saturation Curve")
plt.xlabel("TV Spend (adstocked)")
plt.ylabel("Transformed Effect")
plt.grid(True)
plt.show()
