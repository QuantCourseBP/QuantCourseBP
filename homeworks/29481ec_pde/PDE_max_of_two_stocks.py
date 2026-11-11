# # Assignment 2 - PDE: the contract that pays the better of two stocks
# 
# $S_1(t)$ and $S_2(t)$ are **independent** stocks with **equal volatility** $\sigma$, $S_1(0)=S_2(0)=100$.
# At maturity $T$ the contract pays
# $$V(T) = \max\{S_1(T),\,S_2(T)\}.$$
# 
# * **a.** formula for the fair value
# * **b.** derivation of the PDE
# * **c.** solution of the PDE with the finite difference method
# * **d.** comparison with Monte Carlo
# 
# The assignment does not fix $\sigma$, $T$ and the interest rate $r$, so the base case is $\sigma = 25\%$, $T = 1$, $r = 5\%$ (other values are shown in tables).

import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import norm

plt.rcParams['figure.figsize'] = (7, 4)
plt.rcParams['axes.grid'] = True

S0, R, SIGMA, T = 100.0, 0.05, 0.25, 1.0      # base case

# ## a. Fair value of the contract
# 
# Under the risk-neutral measure $Q$ both stocks follow $dS_i = rS_i\,dt + \sigma S_i\,dW_i$ with independent Brownian motions, and
# 
# $$V(0) = e^{-rT}\,E^Q\big[\max(S_1(T),S_2(T))\big].$$
# 
# **Step 1.** $\max(a,b) = b + (a-b)^+$, hence
# $$V(0) = e^{-rT}E^Q[S_2(T)] + e^{-rT}E^Q\big[(S_1(T)-S_2(T))^+\big].$$
# 
# **Step 2.** The discounted stock price is a martingale (no dividends), so the first term is simply $S_2(0)$.
# 
# **Step 3.** The second term is the price of an option to *exchange* stock 2 for stock 1 (Margrabe). Take $S_2$ as numeraire: with $X = S_1/S_2$,
# $$e^{-rT}E^Q[(S_1(T)-S_2(T))^+] = S_2(0)\,E^{Q_2}\big[(X_T-1)^+\big].$$
# Under $Q_2$ the ratio $X$ is a martingale and lognormal with volatility $\sigma_M=\sqrt{\sigma_1^2+\sigma_2^2-2\rho\sigma_1\sigma_2}=\sigma\sqrt2$ (independence, $\rho=0$). Black's formula with zero interest rate and strike 1 gives $X_0N(d_1)-N(d_2)$.
# 
# **Result.** With $\tau = T-t$, $\sigma_M=\sigma\sqrt2$:
# $$\boxed{V(t,s_1,s_2) = s_2 + s_1N(d_1) - s_2N(d_2)},\qquad d_1=\frac{\ln(s_1/s_2)+\tfrac12\sigma_M^2\tau}{\sigma_M\sqrt\tau},\quad d_2=d_1-\sigma_M\sqrt\tau .$$
# 
# **At $s_1=s_2=100$:** $\ln(s_1/s_2)=0$, so $d_1=\sigma\sqrt{T/2}=-d_2$ and
# $$\boxed{V(0)=100+100\,[N(d_1)-N(-d_1)] = 200\,N\!\left(\sigma\sqrt{T/2}\right)}$$
# 
# Remarks: the value does **not depend on the interest rate** (discounted stocks are martingales); it equals 100 for $T\to0$ or $\sigma\to0$ and grows towards 200 as $\sigma\sqrt T\to\infty$.

def fair_value(s1, s2, tau, sigma):
    """General fair value V(t, s1, s2) of max(S1, S2), tau = T - t."""
    sigma_m = sigma * np.sqrt(2.0)
    d1 = (np.log(s1 / s2) + 0.5 * sigma_m ** 2 * tau) / (sigma_m * np.sqrt(tau))
    d2 = d1 - sigma_m * np.sqrt(tau)
    return s2 + s1 * norm.cdf(d1) - s2 * norm.cdf(d2)


def fair_value_at_100(T, sigma):
    """Special case S1(0) = S2(0) = 100:  200 N(sigma sqrt(T/2))."""
    return 200.0 * norm.cdf(sigma * np.sqrt(T / 2.0))


assert abs(fair_value(100, 100, T, SIGMA) - fair_value_at_100(T, SIGMA)) < 1e-10
print(f'Base case (sigma={SIGMA}, T={T}):  V(0) = {fair_value_at_100(T, SIGMA):.6f}')

sigmas, maturities = [0.1, 0.2, 0.3, 0.4], [0.25, 0.5, 1.0, 2.0]
print('\nFair value V(0) = 200 N(sigma sqrt(T/2))')
print('sigma \\ T ' + ''.join(f'{t:>10}' for t in maturities))
for s in sigmas:
    print(f'{s:>9} ' + ''.join(f'{fair_value_at_100(t, s):10.4f}' for t in maturities))

sig_grid = np.linspace(0.01, 1.0, 200)
for t in maturities[1:]:
    plt.plot(sig_grid, fair_value_at_100(t, sig_grid), label=f'T = {t}')
plt.axhline(100, color='black', linestyle='--', linewidth=1)
plt.title('Fair value of max($S_1$, $S_2$) at $S_1=S_2=100$')
plt.xlabel('volatility $\\sigma$')
plt.ylabel('$V(0)$')
plt.legend()
plt.show()

# ## b. The partial differential equation
# 
# Let $V(t,S_1,S_2)$ be the value of the contract. Because the Brownian motions are independent, $dW_1\,dW_2=0$ and Itô's lemma has **no cross term**:
# $$dV = V_t\,dt + V_{s_1}dS_1 + V_{s_2}dS_2 + \tfrac12\sigma^2\big(S_1^2V_{s_1s_1}+S_2^2V_{s_2s_2}\big)dt .$$
# 
# **Delta hedge.** Build the portfolio $\Pi = V - \Delta_1S_1 - \Delta_2S_2$ with $\Delta_1=V_{s_1}$, $\Delta_2=V_{s_2}$. The random terms cancel:
# $$d\Pi = \Big(V_t + \tfrac12\sigma^2\big(S_1^2V_{s_1s_1}+S_2^2V_{s_2s_2}\big)\Big)dt \quad\text{(riskless)}.$$
# No arbitrage forces $d\Pi = r\Pi\,dt = r\,(V - S_1V_{s_1}-S_2V_{s_2})\,dt$. Therefore
# 
# $$\boxed{\;\frac{\partial V}{\partial t} + \frac{\sigma^2}{2}\Big(s_1^2\frac{\partial^2V}{\partial s_1^2}+s_2^2\frac{\partial^2V}{\partial s_2^2}\Big) + r\Big(s_1\frac{\partial V}{\partial s_1}+s_2\frac{\partial V}{\partial s_2}\Big) - rV = 0\;}$$
# 
# (the Feynman-Kac theorem gives the same equation for $V=e^{-r\tau}E^Q[\dots]$, which is what Monte Carlo estimates in part d).
# 
# **Terminal condition:** $V(T,s_1,s_2)=\max(s_1,s_2)$.
# 
# **Boundary conditions:**
# * $V(t,0,s_2)=s_2$ and $V(t,s_1,0)=s_1$: a worthless stock never wins, the contract equals the other stock (whose value is its price).
# * $V(t,s_1,s_2)\to s_1$ for $s_1\gg s_2$ and $V\to s_2$ for $s_2\gg s_1$.
# 
# **Remark (homogeneity).** $V(t,\lambda s_1,\lambda s_2)=\lambda V(t,s_1,s_2)$. Writing $V=s_2f(t,z)$, $z=s_1/s_2$, the PDE collapses to $f_t+\tfrac12\sigma_M^2z^2f_{zz}=0$ with $\sigma_M^2=2\sigma^2$, which is the Black-Scholes equation with zero rate - consistent with the Margrabe formula of part a.
# 
# **Check:** the formula of part a must satisfy this PDE. We evaluate the residual with finite differences of the closed-form solution (written in $\tau = T-t$, so $V_t=-V_\tau$):

def pde_residual(s1, s2, tau, sigma, r, h=1e-2, ht=1e-4):
    V = lambda a, b, c: fair_value(a, b, c, sigma)
    v = V(s1, s2, tau)
    v_tau = (V(s1, s2, tau + ht) - V(s1, s2, tau - ht)) / (2 * ht)
    v1 = (V(s1 + h, s2, tau) - V(s1 - h, s2, tau)) / (2 * h)
    v2 = (V(s1, s2 + h, tau) - V(s1, s2 - h, tau)) / (2 * h)
    v11 = (V(s1 + h, s2, tau) - 2 * v + V(s1 - h, s2, tau)) / h ** 2
    v22 = (V(s1, s2 + h, tau) - 2 * v + V(s1, s2 - h, tau)) / h ** 2
    return -v_tau + 0.5 * sigma ** 2 * (s1 ** 2 * v11 + s2 ** 2 * v22) + r * (s1 * v1 + s2 * v2) - r * v


print(f"{'s1':>6} {'s2':>6} {'tau':>5} {'PDE residual':>14}")
for s1, s2, tau in [(100, 100, 1.0), (90, 110, 0.5), (130, 80, 0.7), (60, 75, 1.5), (100, 100, 0.1)]:
    res = pde_residual(s1, s2, tau, SIGMA, R)
    print(f'{s1:6.0f} {s2:6.0f} {tau:5.1f} {res:14.2e}')
    assert abs(res) < 1e-5

# terminal condition and boundaries
print('\nterminal condition  V(T, 90, 120) =', fair_value(90, 120, 1e-10, SIGMA).round(6), '(= max = 120)')
print('boundary            V(t, 1e-8, 120) =', fair_value(1e-8, 120, 1.0, SIGMA).round(6), '(= s2 = 120)')

# ## c. Solution of the PDE with the finite difference method
# 
# **Log variables.** With $x_i=\ln s_i$ and time to maturity $\tau=T-t$ the PDE has constant coefficients:
# $$V_\tau = \tfrac12\sigma^2\,(V_{xx}+V_{yy}) + \big(r-\tfrac12\sigma^2\big)(V_x+V_y) - rV,\qquad V(\tau=0)=\max(e^x,e^y).$$
# 
# **Scheme**
# * uniform grid in $(x,y)$ on $[\ln 100 \pm 6\sigma\sqrt T]^2$ ($\ln 100$ is a grid node), central differences for $V_x,V_{xx}$ (second order),
# * Dirichlet boundary values $V=\max(s_1,s_2)$ on the four edges (justified by the boundary conditions of part b; their error does not reach the centre of the domain),
# * time stepping with **Crank-Nicolson** (second order), started with four implicit Euler half steps (**Rannacher start-up**) because the payoff has a kink along $s_1=s_2$,
# * the sparse linear system $(I-\tfrac{\Delta\tau}{2}A)V^{k+1}=\dots$ has the same matrix in every step, so its LU factorisation is computed once.

def solve_pde(sigma, r, T, s0=100.0, n=161, n_time=None, width=6.0):
    """Crank-Nicolson finite difference solution of the 2D PDE in log variables.

    n: number of grid points per direction (odd, so that ln(s0) is a node)
    returns the log-grid x and the array V[i, j] = V(t=0, s1=exp(x[i]), s2=exp(x[j]))
    """
    half = width * sigma * np.sqrt(T)
    x0 = np.log(s0)
    x = np.linspace(x0 - half, x0 + half, n)
    dx = x[1] - x[0]
    n_time = n_time or n
    dt = T / n_time
    mu = r - 0.5 * sigma ** 2
    m = n - 2                                                   # interior points per direction

    # 1D operator (second derivative + drift) on interior nodes; then the 2D operator via Kronecker sums
    a = 0.5 * sigma ** 2 / dx ** 2
    lower, diag, upper = a - mu / (2 * dx), -2 * a, a + mu / (2 * dx)
    D = sp.diags([np.full(m - 1, lower), np.full(m, diag), np.full(m - 1, upper)], [-1, 0, 1], format='csr')
    I = sp.identity(m, format='csr')
    A = (sp.kron(D, I) + sp.kron(I, D) - r * sp.identity(m * m)).tocsc()

    # initial condition (payoff) and Dirichlet boundary contribution b
    S1, S2 = np.meshgrid(np.exp(x), np.exp(x), indexing='ij')
    V = np.maximum(S1, S2)
    b = np.zeros((m, m))
    b[0, :] += lower * V[0, 1:-1]
    b[-1, :] += upper * V[-1, 1:-1]
    b[:, 0] += lower * V[1:-1, 0]
    b[:, -1] += upper * V[1:-1, -1]
    b = b.ravel()
    u = V[1:-1, 1:-1].ravel()

    Id = sp.identity(m * m, format='csc')
    lu = spla.splu((Id - 0.5 * dt * A).tocsc())                 # same matrix for Euler half steps and CN steps
    for _ in range(4):                                          # Rannacher: 4 implicit Euler steps of size dt/2
        u = lu.solve(u + 0.5 * dt * b)
    explicit_part = (Id + 0.5 * dt * A).tocsr()
    for _ in range(n_time - 2):                                 # Crank-Nicolson steps
        u = lu.solve(explicit_part @ u + dt * b)

    V[1:-1, 1:-1] = u.reshape(m, m)
    return x, V


def pde_value_at_s0(x, V):
    c = len(x) // 2                                             # index of ln(s0)
    return V[c, c]


t0 = time.time()
x, V_pde = solve_pde(SIGMA, R, T, n=201)
exact = fair_value_at_100(T, SIGMA)
print(f'PDE (201 x 201 grid, {time.time() - t0:.1f}s): V(0,100,100) = {pde_value_at_s0(x, V_pde):.6f}')
print(f'Exact formula:                       V(0,100,100) = {exact:.6f}')
print(f'Error: {pde_value_at_s0(x, V_pde) - exact:.2e}')

# the whole solution surface V(0, s1, s2) and its error against the closed-form solution
s = np.exp(x)
S1g, S2g = np.meshgrid(s, s, indexing='ij')
V_exact = fair_value(S1g, S2g, T, SIGMA)
window = (S1g > 50) & (S1g < 200) & (S2g > 50) & (S2g < 200)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
levels = np.linspace(50, 200, 16)
c1 = axes[0].contourf(S1g, S2g, V_pde, levels=levels, extend='both', cmap='viridis')
axes[0].set_xlim(50, 200); axes[0].set_ylim(50, 200)
axes[0].set_title('PDE solution $V(0,s_1,s_2)$'); axes[0].set_xlabel('$s_1$'); axes[0].set_ylabel('$s_2$')
plt.colorbar(c1, ax=axes[0])
axes[0].plot([50, 200], [50, 200], 'w--', linewidth=1)

err = np.where(window, V_pde - V_exact, np.nan)
c2 = axes[1].pcolormesh(S1g, S2g, err, cmap='coolwarm', shading='auto',
                        vmin=-np.nanmax(np.abs(err)), vmax=np.nanmax(np.abs(err)))
axes[1].set_xlim(50, 200); axes[1].set_ylim(50, 200)
axes[1].set_title('PDE minus exact formula'); axes[1].set_xlabel('$s_1$'); axes[1].set_ylabel('$s_2$')
plt.colorbar(c2, ax=axes[1])

sd = np.linspace(50, 200, 100)
diag_idx = np.argmin(np.abs(s[:, None] - sd[None, :]), axis=0)
axes[2].plot(s[(s > 50) & (s < 200)], np.diag(V_pde)[(s > 50) & (s < 200)], label='PDE, $s_1=s_2$', linewidth=3)
axes[2].plot(sd, 2 * norm.cdf(SIGMA * np.sqrt(T / 2)) * sd, '--', label='exact $2N(\\sigma\\sqrt{T/2})\\,s$')
axes[2].plot(sd, sd, ':', color='gray', label='payoff $\\max(s_1,s_2)=s$')
axes[2].set_title('Cross-section along $s_1=s_2$'); axes[2].set_xlabel('$s$'); axes[2].legend()
plt.tight_layout()
plt.show()
print(f'max |error| over 50 < s1, s2 < 200: {np.nanmax(np.abs(err)):.3e}')

# **Convergence.** Refine the grid (and the time step with it). The error at $(100,100)$ should drop by a factor 4 each time the step is halved (second order).

exact = fair_value_at_100(T, SIGMA)
rows = []
print(f"{'grid n':>7} {'step dx':>9} {'V(0,100,100)':>14} {'error':>11} {'order':>7} {'time[s]':>8}")
previous = None
for n in [41, 81, 161, 321]:
    t0 = time.time()
    xg, Vg = solve_pde(SIGMA, R, T, n=n)
    elapsed = time.time() - t0
    dx = xg[1] - xg[0]
    err_n = abs(pde_value_at_s0(xg, Vg) - exact)
    order = '' if previous is None else f'{np.log2(previous / err_n):7.2f}'
    print(f'{n:7d} {dx:9.5f} {pde_value_at_s0(xg, Vg):14.6f} {err_n:11.2e} {order:>7} {elapsed:8.1f}')
    rows.append((dx, err_n))
    previous = err_n

dxs, errs = np.array(rows).T
plt.loglog(dxs, errs, 'o-', label='PDE error at (100, 100)')
plt.loglog(dxs, errs[0] * (dxs / dxs[0]) ** 2, '--', color='gray', label='slope 2')
plt.xlabel('grid step $\\Delta x$')
plt.ylabel('absolute error')
plt.title('Convergence of the Crank-Nicolson solution')
plt.legend()
plt.show()

# ## d. Comparison with Monte Carlo
# 
# Under $Q$ the stocks can be simulated exactly,
# $$S_i(T)=S_i(0)\exp\!\big((r-\tfrac12\sigma^2)T+\sigma\sqrt T\,Z_i\big),\quad Z_1,Z_2\ \text{independent } N(0,1),$$
# and $V(0)\approx e^{-rT}\,\frac1N\sum\max(S_1^{(k)}(T),S_2^{(k)}(T))$ with standard error $s/\sqrt N$.
# 
# Three estimators:
# 1. **plain** Monte Carlo,
# 2. **antithetic variates** ($Z\to-Z$),
# 3. **control variate:** $e^{-rT}\frac{S_1(T)+S_2(T)}2$ has the exactly known mean $100$ (martingale), and it is strongly correlated with the payoff, because $\max(a,b)=\frac{a+b}2+\frac{|a-b|}2$. The coefficient is estimated by regression.

def mc_price(sigma, r, T, s0=100.0, n_paths=1_000_000, method='plain', seed=0):
    """Monte Carlo price of max(S1, S2). Returns (estimate, standard error)."""
    rng = np.random.default_rng(seed)
    drift, vol = (r - 0.5 * sigma ** 2) * T, sigma * np.sqrt(T)
    discount = np.exp(-r * T)
    z = rng.standard_normal((2, n_paths))

    def discounted(z):
        s = s0 * np.exp(drift + vol * z)
        return discount * np.maximum(s[0], s[1]), discount * 0.5 * (s[0] + s[1])

    payoff, average = discounted(z)
    if method == 'plain':
        sample = payoff
    elif method == 'antithetic':
        payoff_anti, _ = discounted(-z)
        sample = 0.5 * (payoff + payoff_anti)
    elif method == 'control':
        beta = np.cov(payoff, average)[0, 1] / np.var(average, ddof=1)
        sample = payoff - beta * (average - s0)              # E[average] = s0 exactly
    else:
        raise ValueError(method)
    return sample.mean(), sample.std(ddof=1) / np.sqrt(n_paths)


exact = fair_value_at_100(T, SIGMA)
pde = pde_value_at_s0(*solve_pde(SIGMA, R, T, n=321))
print(f'Base case sigma={SIGMA}, T={T}, r={R}, 1 000 000 paths')
print(f'  exact formula : {exact:.5f}')
print(f'  PDE (321x321) : {pde:.5f}   error {pde - exact:+.5f}\n')
print(f"{'method':<12} {'estimate':>10} {'std error':>10} {'95% interval':>24} {'error':>9}  exact inside?")
for method in ['plain', 'antithetic', 'control']:
    est, se = mc_price(SIGMA, R, T, method=method)
    lo, hi = est - 1.96 * se, est + 1.96 * se
    print(f'{method:<12} {est:10.4f} {se:10.4f}   [{lo:9.4f}, {hi:9.4f}] {est - exact:+9.4f}  {lo <= exact <= hi}')

# **Convergence of Monte Carlo:** root-mean-square error over 40 independent runs for increasing numbers of paths. The error falls like $1/\sqrt N$; the control variate gives the same slope with a much smaller constant.

paths = [10 ** 3, 3 * 10 ** 3, 10 ** 4, 3 * 10 ** 4, 10 ** 5, 3 * 10 ** 5, 10 ** 6]
rms = {'plain': [], 'control': []}
for n_paths in paths:
    for method in rms:
        errors = [mc_price(SIGMA, R, T, n_paths=n_paths, method=method, seed=100 + k)[0] - exact for k in range(40)]
        rms[method].append(np.sqrt(np.mean(np.square(errors))))

for method, values in rms.items():
    plt.loglog(paths, values, 'o-', label=f'Monte Carlo ({method})')
plt.loglog(paths, rms['plain'][0] * np.sqrt(paths[0] / np.array(paths)), '--', color='gray', label='$1/\\sqrt{N}$')
plt.axhline(abs(pde - exact), color='tab:red', linestyle=':', label='PDE error (321 x 321 grid)')
plt.xlabel('number of paths $N$')
plt.ylabel('RMS error of $V(0)$')
plt.title('Monte Carlo convergence vs PDE accuracy')
plt.legend()
plt.show()
print('variance reduction of the control variate at 1 000 000 paths: factor',
      round((rms['plain'][-1] / rms['control'][-1]) ** 2, 1))

# **Other parameters:** the three methods against the exact formula for several volatilities and maturities (PDE on a 161 x 161 grid, Monte Carlo with control variate and 1 000 000 paths).

print(f"{'sigma':>6} {'T':>5} {'exact':>10} {'PDE':>10} {'PDE err':>9} {'MC (CV)':>10} {'MC std err':>11} {'MC err':>9}")
for sigma, T_ in [(0.10, 0.5), (0.20, 1.0), (0.25, 1.0), (0.40, 2.0), (0.60, 1.0)]:
    ex = fair_value_at_100(T_, sigma)
    pd_val = pde_value_at_s0(*solve_pde(sigma, R, T_, n=161))
    mc, se = mc_price(sigma, R, T_, method='control')
    print(f'{sigma:6.2f} {T_:5.1f} {ex:10.4f} {pd_val:10.4f} {pd_val - ex:+9.4f} {mc:10.4f} {se:11.4f} {mc - ex:+9.4f}')

# ## Summary
# 
# * **a.** $V(0)=200\,N\big(\sigma\sqrt{T/2}\big)$ for $S_1(0)=S_2(0)=100$ (general case: $V=s_2+s_1N(d_1)-s_2N(d_2)$ with $\sigma_M=\sigma\sqrt2$). It does not depend on $r$.
# * **b.** $V_t+\tfrac{\sigma^2}2(s_1^2V_{s_1s_1}+s_2^2V_{s_2s_2})+r(s_1V_{s_1}+s_2V_{s_2})-rV=0$, $V(T)=\max(s_1,s_2)$; no cross term because of independence. The closed-form solution satisfies it (residuals of order $10^{-6}$ or smaller, which is the rounding level of the finite differences used for the check).
# * **c.** The Crank-Nicolson / Rannacher scheme in log variables converges with second order; the 321 x 321 grid is accurate to about $10^{-3}$ on a value of about 114.
# * **d.** Monte Carlo agrees with the PDE and the exact formula within its confidence interval. The PDE error is deterministic and shrinks like $\Delta x^2$, while Monte Carlo's error is random and shrinks like $1/\sqrt N$ (even with $10^6$ paths and the control variate its error, about 0.01, is ten times larger than that of the 321 x 321 PDE grid, about 0.001; matching it would take about $10^8$ paths). In two dimensions the PDE is much more accurate for the cost; for many underlyings the grid becomes impossible (curse of dimensionality) and Monte Carlo is the practical method.
