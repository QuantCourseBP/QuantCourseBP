import numpy as np
import pytest

from enums import (
    Stock, PutCallFwd, LongShort, GreekMethod,
    FiniteDifferenceMethod, BumpType
)
from market_data import MarketData, VolGrid
from model import FlatVolModel
from contract import EuropeanContract
from pricer import EuropeanAnalyticPricer, Params


@pytest.fixture
def market(monkeypatch):
    """Controlled market: spot 100, volatility 20%, rate 5%."""
    stock = Stock.BLUECHIP_BANK

    points = np.array([
        [80., 0.5],
        [120., 0.5],
        [80., 2.0],
        [120., 2.0]
    ])

    grid = VolGrid(stock, points, np.full(4, 0.20))

    monkeypatch.setattr(MarketData, "spot", {stock: 100.})
    monkeypatch.setattr(MarketData, "volgrid", {stock: grid})
    monkeypatch.setattr(MarketData, "risk_free_rate", 0.05)
    monkeypatch.setattr(MarketData, "is_initialized", True)

    return stock


@pytest.mark.parametrize("fd_method", list(FiniteDifferenceMethod))
@pytest.mark.parametrize("bump_type", list(BumpType))
@pytest.mark.parametrize(
    "option_type", [PutCallFwd.CALL, PutCallFwd.PUT]
)
def test_numerical_greeks(market, fd_method, bump_type, option_type):
    model = FlatVolModel(market)

    contract = EuropeanContract(
        market, option_type, LongShort.LONG, 100., 1.
    )

    pricer = EuropeanAnalyticPricer(contract, model, Params())

    pricer.fd_method = fd_method
    pricer.bump_type = bump_type

    # Both conventions produce the same effective spot change: h = 0.01.
    pricer.absolute_bump_size = 0.01
    pricer.relative_bump_size = 0.0001

    # Check that bumping preserves the valuation time.
    pricer.valuation_time = 0.2

    delta_ref = pricer.calc_delta(GreekMethod.ANALYTIC)
    gamma_ref = pricer.calc_gamma(GreekMethod.ANALYTIC)

    delta_num = pricer.calc_delta(GreekMethod.BUMP)
    gamma_num = pricer.calc_gamma(GreekMethod.BUMP)

    assert delta_num == pytest.approx(delta_ref, abs=3e-4)
    assert gamma_num == pytest.approx(gamma_ref, abs=1e-5)

    # The original model must remain unchanged.
    assert model.spot == 100.
    assert pricer.valuation_time == 0.2