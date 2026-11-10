"""Pricing identities, numerical convergence, boundary and sampling checks."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm
from .two_stock import analytic_price, finite_difference_price, monte_carlo_price


def test_equal_spot_closed_form():
    assert analytic_price() == pytest.approx(200*norm.cdf(.2/np.sqrt(2)), abs=1e-12)


@pytest.mark.parametrize('spot1,spot2,vol,tau', [(100,100,.2,1), (80,120,.3,.5), (150,75,.4,2)])
def test_analytic_against_independent_integral(spot1, spot2, vol, tau):
    # E[max(X,Y)] = E[X] + integral_0^infinity P(Y>z) P(X<z) dz.
    # X,Y here are already discounted lognormal stock prices.
    scale = vol*np.sqrt(tau)
    def integrand(z):
        if z == 0:
            return 0.
        x = (np.log(z/spot1)+.5*scale**2)/scale
        y = (np.log(z/spot2)+.5*scale**2)/scale
        return norm.cdf(x)*norm.sf(y)
    integrated = spot1 + quad(integrand, 0, np.inf, epsabs=1e-8)[0]
    assert analytic_price(spot1,spot2,vol,tau) == pytest.approx(integrated,rel=1e-9)


@pytest.mark.parametrize('vol,tau', [(0,1),(.2,0),(0,0)])
def test_deterministic_limits(vol,tau):
    expected = 120.
    assert analytic_price(80,120,vol,tau) == expected
    assert finite_difference_price(80,120,vol,tau,space_steps=10).price == expected
    mc = monte_carlo_price(80,120,vol,tau,num_paths=100)
    assert mc.price == expected and mc.standard_error == 0.


@pytest.mark.parametrize('spot1,spot2', [(100,100),(80,120),(130,70)])
def test_symmetry_homogeneity_and_bounds(spot1,spot2):
    a = analytic_price(spot1,spot2)
    assert a == analytic_price(spot2,spot1)
    assert analytic_price(3*spot1,3*spot2) == pytest.approx(3*a)
    assert max(spot1,spot2) <= a <= spot1+spot2
    p = finite_difference_price(spot1,spot2,space_steps=40,time_steps=80)
    swapped = finite_difference_price(spot2,spot1,space_steps=40,time_steps=80)
    scaled = finite_difference_price(3*spot1,3*spot2,space_steps=40,time_steps=80)
    assert swapped.price == pytest.approx(p.price,abs=1e-10)
    np.testing.assert_allclose(swapped.values,p.values.T,rtol=1e-12)
    assert scaled.price == pytest.approx(3*p.price,rel=1e-12)
    assert max(spot1,spot2) <= p.price <= spot1+spot2


@pytest.mark.parametrize('vol,tau,rate', [(.15,.5,.03),(.2,1,.05),(.3,2,0),(.25,1,-.01)])
def test_pde_against_analytic(vol,tau,rate):
    p = finite_difference_price(volatility=vol,maturity=tau,rate=rate,
                                space_steps=160,time_steps=320,log_half_width=2.)
    assert abs(p.price-analytic_price(volatility=vol,maturity=tau,rate=rate)) < .035


def test_spatial_refinement():
    exact = analytic_price()
    errors = [abs(finite_difference_price(space_steps=n,time_steps=400).price-exact)
              for n in [40,80,160]]
    assert errors[2] < errors[1] < errors[0]
    assert 3 < errors[0]/errors[1] < 5
    assert 3 < errors[1]/errors[2] < 5


def test_domain_boundary_and_terminal():
    small = finite_difference_price(space_steps=20,time_steps=200,log_half_width=.3)
    medium = finite_difference_price(space_steps=60,time_steps=200,log_half_width=.9)
    large = finite_difference_price(space_steps=100,time_steps=200,log_half_width=1.5)
    assert abs(medium.price-large.price) < 1e-4
    assert abs(small.price-large.price) > .01
    boundary = np.maximum(large.spot1_grid[:,None],large.spot2_grid[None,:])
    np.testing.assert_array_equal(large.values[0,:],boundary[0,:])
    np.testing.assert_array_equal(large.values[-1,:],boundary[-1,:])
    np.testing.assert_array_equal(large.values[:,0],boundary[:,0])
    np.testing.assert_array_equal(large.values[:,-1],boundary[:,-1])
    terminal = finite_difference_price(maturity=0,space_steps=20)
    np.testing.assert_array_equal(terminal.values,
                                 np.maximum(terminal.spot1_grid[:,None],terminal.spot2_grid[None,:]))


def test_rate_independence_and_volatility_monotonicity():
    assert analytic_price(rate=-.05) == analytic_price(rate=.15)
    prices = [analytic_price(volatility=v) for v in [0,.1,.2,.4]]
    assert np.all(np.diff(prices)>0)
    low_rate = finite_difference_price(rate=0,space_steps=100,time_steps=200).price
    high_rate = finite_difference_price(rate=.1,space_steps=100,time_steps=200).price
    assert abs(low_rate-high_rate) < .015


@pytest.mark.parametrize('spot1,spot2,vol,tau', [(100,100,.2,1),(80,120,.3,.5),(150,75,.4,2)])
def test_mc_against_analytic(spot1,spot2,vol,tau):
    mc = monte_carlo_price(spot1,spot2,vol,tau,num_paths=200_000,seed=2026)
    assert abs(mc.price-analytic_price(spot1,spot2,vol,tau)) < 4*mc.standard_error
    assert mc.ci_high-mc.price == pytest.approx(1.96*mc.standard_error)
    assert mc.price-mc.ci_low == pytest.approx(1.96*mc.standard_error)


def test_mc_exact_draws_and_standard_error():
    n,seed,s1,s2,vol,tau,rate = 1000,42,80.,120.,.3,.5,.07
    z = np.random.default_rng(seed).standard_normal((n,2))
    stocks = np.array([s1,s2])*np.exp((rate-.5*vol**2)*tau+vol*np.sqrt(tau)*z)
    discounted = np.exp(-rate*tau)*np.max(stocks,axis=1)
    mc = monte_carlo_price(s1,s2,vol,tau,rate,n,seed)
    assert mc.price == pytest.approx(discounted.mean(),rel=1e-14)
    assert mc.standard_error == pytest.approx(discounted.std(ddof=1)/np.sqrt(n),rel=1e-14)
    assert monte_carlo_price(s1,s2,vol,tau,rate,n,seed).price == mc.price
    assert monte_carlo_price(s1,s2,vol,tau,0.,n,seed).price == mc.price


@pytest.mark.parametrize('kwargs', [{'spot1':0},{'spot2':-1},{'volatility':-.1},
                                  {'maturity':-1},{'rate':np.nan},{'spot1':np.inf}])
@pytest.mark.parametrize('method',[analytic_price,finite_difference_price,monte_carlo_price])
def test_invalid_model_inputs(method,kwargs):
    with pytest.raises(ValueError):
        method(**kwargs)


@pytest.mark.parametrize('kwargs',[{'space_steps':5},{'space_steps':2},{'space_steps':10.5},
                                  {'time_steps':0},{'time_steps':True},
                                  {'log_half_width':0},{'log_half_width':np.inf},
                                  {'volatility':.001,'rate':.2,'space_steps':10}])
def test_invalid_pde_grid(kwargs):
    with pytest.raises(ValueError):
        finite_difference_price(**kwargs)


@pytest.mark.parametrize('kwargs',[{'num_paths':1},{'num_paths':10.5},{'seed':-1},{'seed':True}])
def test_invalid_mc_configuration(kwargs):
    with pytest.raises(ValueError):
        monte_carlo_price(**kwargs)
