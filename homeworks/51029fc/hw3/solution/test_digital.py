
import numpy as np
import pytest
from scipy.stats import norm

from pricer import *
from contract import *
from model import *


@pytest.mark.parametrize('underlying', [
    Stock.BLUECHIP_BANK,
    Stock.TIPTOP_SOLUTIONS
])
@pytest.mark.parametrize('strike_over_spot', [0.5, 1.0, 1.5])
@pytest.mark.parametrize('expiry', [0.5, 1.0, 1.5, 2.0])
@pytest.mark.parametrize('derivative_type', [
    PutCallFwd.CALL,
    PutCallFwd.PUT
])
def test_digital_fair_value(
    underlying, strike_over_spot, expiry, derivative_type
):
    model = MarketModel(underlying)
    strike = strike_over_spot * model.spot

    contract = EuropeanDigitalContract(
        underlying,
        derivative_type,
        LongShort.LONG,
        strike,
        expiry
    )

    pricer = EuropeanDigitalAnalyticPricer(
        contract, model, Params()
    )

    # Actual price from our implementation
    actual = pricer.calc_fair_value()

    # Expected price from Black-Scholes formula
    S = model.spot
    K = strike
    T = expiry
    r = model.risk_free_rate
    vol = model.vol

    d2 = (
        np.log(S / K) + (r - vol**2 / 2) * T
    ) / (vol * np.sqrt(T))

    df = np.exp(-r * T)

    if derivative_type == PutCallFwd.CALL:
        expected = df * norm.cdf(d2)
    else:
        expected = df * norm.cdf(-d2)

    assert actual == pytest.approx(expected, abs=1e-6)
