import numpy as np
import pytest
from scipy.stats import binom, norm
from tree_calibration import (calibrate, balanced_call_price, calcBalancedDownStep,
                              create_spot_tree, create_discounted_price_tree)


def bsm_call(spot, K, expiry, rate, vol):
    d1 = (np.log(spot/K) + (rate + .5*vol**2)*expiry)/(vol*np.sqrt(expiry))
    d2 = d1 - vol*np.sqrt(expiry)
    return spot*norm.cdf(d1) - K*np.exp(-rate*expiry)*norm.cdf(d2)


@pytest.mark.parametrize('steps', [1, 2, 5, 16, 64])
@pytest.mark.parametrize('rate', [-.02, 0., .05])
def test_calibration_recovers_generated_price_and_step(steps, rate):
    spot, K, expiry = 100., 105., 1.3
    growth = np.exp(rate*expiry/steps)
    expected_up = growth*(1 + .25/np.sqrt(steps))
    observed = balanced_call_price(spot, expected_up, steps, K, expiry, rate)
    up = calibrate(observed, steps, K, expiry, rate, spot)
    assert up == pytest.approx(expected_up, abs=1e-11)
    assert balanced_call_price(spot, up, steps, K, expiry, rate) == pytest.approx(observed, abs=1e-10)
    df = np.exp(-rate*expiry/steps)
    down = calcBalancedDownStep(up, df)
    assert 0 < down < growth < up
    assert (1/df-down)/(up-down) == pytest.approx(.5)
    assert df**steps == pytest.approx(np.exp(-rate*expiry))


@pytest.mark.parametrize('steps', [1, 2, 8, 32])
@pytest.mark.parametrize('spot', [1., 100.])
def test_price_against_independent_binomial_expectation(steps, spot):
    expiry, rate = 1.7, .03
    K = 1.05*spot
    df = np.exp(-rate*expiry/steps)
    up = (1/df)*(1+.3/np.sqrt(steps))
    down = 2/df-up
    j = np.arange(steps+1)
    leaves = spot*up**j*down**(steps-j)
    expected = np.exp(-rate*expiry)*np.sum(binom.pmf(j, steps, .5)*np.maximum(leaves-K, 0))
    assert balanced_call_price(spot, up, steps, K, expiry, rate) == pytest.approx(expected)
    price_tree = create_discounted_price_tree(create_spot_tree(spot, up, down, steps), df, K)
    assert price_tree[0][0] == pytest.approx(expected)


@pytest.mark.parametrize('K', [80., 100., 120.])
@pytest.mark.parametrize('expiry', [.25, 1., 2.])
def test_one_step_calibration_against_closed_form(K, expiry):
    spot, rate = 100., .05
    observed = bsm_call(spot, K, expiry, rate, .25)
    up = calibrate(observed, 1, K, expiry, rate, spot)
    df = np.exp(-rate*expiry)
    # At a non-boundary one-step solution the down call expires worthless.
    expected_up = (K+2*observed/df)/spot
    assert up == pytest.approx(expected_up, abs=1e-11)
    down = calcBalancedDownStep(up, df)
    assert spot*down < K < spot*up
    assert .5*df*(spot*up-K) == pytest.approx(observed)


@pytest.mark.parametrize('steps', [1, 2, 4, 8, 16, 32, 64, 128])
def test_same_bsm_target_across_step_counts(steps):
    target = bsm_call(100., 100., 1., .05, .2)
    up = calibrate(target, steps, 100., 1., .05, 100.)
    assert balanced_call_price(100., up, steps, 100., 1., .05) == pytest.approx(target, abs=1e-10)


def test_general_lecture_pricer_uses_correct_probabilities_and_spot_normalization():
    spot, up, down, df, K = 100., 1.3, .9, .98, 100.
    p = (1/df-down)/(up-down)
    expected = df**2*(p**2*max(spot*up**2-K, 0)
                      + 2*p*(1-p)*max(spot*up*down-K, 0)
                      + (1-p)**2*max(spot*down**2-K, 0))
    tree = create_spot_tree(spot, up, down, 2)
    assert create_discounted_price_tree(tree, df, K)[0][0] == pytest.approx(expected)


@pytest.mark.parametrize('change', [dict(steps=0), dict(steps=1.5), dict(steps=True),
    dict(expiry=0), dict(spot=-100), dict(K=0), dict(rate=np.nan), dict(observed_price=np.inf)])
def test_invalid_contract_inputs(change):
    args = dict(observed_price=10., steps=2, K=100., expiry=1., rate=.05, spot=100.)
    args.update(change)
    with pytest.raises(ValueError):
        calibrate(**args)


@pytest.mark.parametrize('price', [-1., 1., 60., 101.])
def test_unattainable_prices(price):
    with pytest.raises(ValueError, match='outside attainable range'):
        calibrate(price, 1, 100., 1., .05, 100.)


def test_deterministic_minimum_is_not_identifiable():
    minimum = 100.-100.*np.exp(-.05)
    with pytest.raises(ValueError, match='not uniquely identifiable'):
        calibrate(minimum, 1, 100., 1., .05, 100.)


def test_zero_down_boundary_is_reported():
    maximum = 100.-.5*100.*np.exp(-.05)
    with pytest.raises(ValueError, match='upper boundary'):
        calibrate(maximum, 1, 100., 1., .05, 100.)
