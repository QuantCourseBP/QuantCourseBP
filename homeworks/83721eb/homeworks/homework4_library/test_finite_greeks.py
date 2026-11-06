
import pytest
from market_data import MarketData
import numpy as np
from market_data import MarketData, VolGrid

# Synthetic market data for testing finite differences only

@pytest.fixture(autouse=True)
def setup_test_market_data(monkeypatch):
    underlying = Stock.BLUECHIP_BANK

    points = np.array([
        [80.0, 0.5],
        [120.0, 0.5],
        [80.0, 2.0],
        [120.0, 2.0]
    ])

    volatilities = np.array([
        0.20, 0.20, 0.20, 0.20
    ])

    test_volgrid = VolGrid(
        underlying, points, volatilities
    )

    monkeypatch.setattr(
        MarketData, "spot", {underlying: 100.0}
    )
    monkeypatch.setattr(
        MarketData, "volgrid", {underlying: test_volgrid}
    )
    monkeypatch.setattr(
        MarketData, "is_initialized", True
    )

from enums import (
    Stock, PutCallFwd, LongShort,
    GreekMethod, FiniteMethod, BumpSizeMethod
)
from contract import EuropeanContract
from model import BSVolModel
from pricer import EuropeanAnalyticPricer, Params


# Homework 4 - Test all finite difference combinations

@pytest.mark.parametrize(
    "option_type",
    [PutCallFwd.CALL, PutCallFwd.PUT]
)
@pytest.mark.parametrize("finite_method", list(FiniteMethod))
@pytest.mark.parametrize("bump_method", list(BumpSizeMethod))
def test_numerical_delta_gamma(
    option_type, finite_method, bump_method
):
    underlying = Stock.BLUECHIP_BANK
    model = BSVolModel(underlying)
    params = Params()

    contract = EuropeanContract(
        underlying,
        option_type,
        LongShort.LONG,
        model.spot,
        1.0
    )

    pricer = EuropeanAnalyticPricer(contract, model, params)

    pricer.finitemethod = finite_method
    pricer.bumpsizemethod = bump_method

    # Use the same effective spot bump for both methods
    h = 0.1

    if bump_method == BumpSizeMethod.ABSOLUTE:
        pricer.relative_bump_size = h
    else:
        pricer.relative_bump_size = h / model.spot

    # Analytical values
    analytic_delta = pricer.calc_delta(GreekMethod.ANALYTIC)
    analytic_gamma = pricer.calc_gamma(GreekMethod.ANALYTIC)

    # Finite difference values
    numerical_delta = pricer.calc_delta(GreekMethod.BUMP)
    numerical_gamma = pricer.calc_gamma(GreekMethod.BUMP)

    # Compare the results
    assert numerical_delta == pytest.approx(
        analytic_delta, rel=0.01, abs=1e-4
    )

    assert numerical_gamma == pytest.approx(
        analytic_gamma, rel=0.02, abs=1e-5
    )
