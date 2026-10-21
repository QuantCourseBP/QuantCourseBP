"""Independent integration checks of digital values and finite-difference Greeks."""
from copy import deepcopy
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm
from .pricer import EuropeanDigitalAnalyticPricer, Params
from .contract import EuropeanDigitalContract, EuropeanContract
from .model import MarketModel
from .enums import Stock, PutCallFwd, LongShort


def make_pricer(kind=PutCallFwd.CALL, position=LongShort.LONG, ratio=1., expiry=1., underlying=Stock.BLUECHIP_BANK):
    model = MarketModel(underlying)
    contract = EuropeanDigitalContract(underlying, kind, position, ratio * model.spot, expiry)
    return EuropeanDigitalAnalyticPricer(contract, model, Params())


@pytest.mark.parametrize("underlying", list(Stock))
@pytest.mark.parametrize("ratio", [.8, 1., 1.2])
@pytest.mark.parametrize("expiry", [.25, 1., 2.])
@pytest.mark.parametrize("kind", [PutCallFwd.CALL, PutCallFwd.PUT])
@pytest.mark.parametrize("position", [LongShort.LONG, LongShort.SHORT])
def test_fair_value_by_normal_integration(underlying, ratio, expiry, kind, position):
    p = make_pricer(kind, position, ratio, expiry, underlying)
    m, c = p.model, p.contract
    # Integrate the risk-neutral probability above/below the exercise threshold.
    z = (np.log(c.strike / m.spot) - (m.risk_free_rate - m.vol**2 / 2) * expiry) / (m.vol * np.sqrt(expiry))
    bounds = (z, np.inf) if kind == PutCallFwd.CALL else (-np.inf, z)
    probability = quad(norm.pdf, *bounds, epsabs=1e-12)[0]
    expected = c.direction * np.exp(-m.risk_free_rate * expiry) * probability
    assert p.calc_fair_value() == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize("kind", [PutCallFwd.CALL, PutCallFwd.PUT])
@pytest.mark.parametrize("position", [LongShort.LONG, LongShort.SHORT])
def test_known_value(kind, position):
    p = make_pricer(kind, position)
    p.model.spot, p.model.vol, p.model.risk_free_rate = 100., .2, .05
    p.contract.strike = 100.
    expected = .5323248154537634 if kind == PutCallFwd.CALL else .41890460904695065
    assert p.calc_fair_value() == pytest.approx(p.contract.direction * expected, abs=1e-12)


@pytest.mark.parametrize("kind", [PutCallFwd.CALL, PutCallFwd.PUT])
@pytest.mark.parametrize("position", [LongShort.LONG, LongShort.SHORT])
@pytest.mark.parametrize("ratio", [.8, 1., 1.2])
def test_greeks_against_central_differences(kind, position, ratio):
    p = make_pricer(kind, position, ratio)
    def bumped(attr, h):
        other = deepcopy(p)
        setattr(other.model, attr, getattr(other.model, attr) + h)
        return other.calc_fair_value()
    for greek, attr, h in [("delta", "spot", .001), ("vega", "vol", 1e-5), ("rho", "risk_free_rate", 1e-5)]:
        numerical = (bumped(attr, h) - bumped(attr, -h)) / (2*h)
        assert getattr(p, "calc_"+greek)() == pytest.approx(numerical, rel=2e-6, abs=1e-8)
    h = .02
    gamma = (bumped("spot", h) - 2*p.calc_fair_value() + bumped("spot", -h)) / h**2
    assert p.calc_gamma() == pytest.approx(gamma, rel=2e-5, abs=1e-8)
    plus, minus = deepcopy(p), deepcopy(p)
    plus.valuation_time, minus.valuation_time = 1e-5, -1e-5
    theta = (plus.calc_fair_value() - minus.calc_fair_value()) / 2e-5
    assert p.calc_theta() == pytest.approx(theta, rel=2e-6, abs=1e-8)


@pytest.mark.parametrize("kind", [PutCallFwd.CALL, PutCallFwd.PUT])
@pytest.mark.parametrize("position", [LongShort.LONG, LongShort.SHORT])
@pytest.mark.parametrize("terminal", [90., 100., 110.])
def test_payoff(kind, position, terminal):
    p = make_pricer(kind, position)
    p.contract.strike = 100.
    expected = terminal > 100 if kind == PutCallFwd.CALL else terminal < 100
    assert p.contract.payoff({1.: terminal}) == p.contract.direction * float(expected)


def test_digital_parity():
    c, p = make_pricer(), make_pricer(PutCallFwd.PUT)
    assert c.calc_fair_value() + p.calc_fair_value() == pytest.approx(c.model.calc_df(1.))
    for greek in ["delta", "gamma", "vega"]:
        assert getattr(c, "calc_"+greek)() + getattr(p, "calc_"+greek)() == pytest.approx(0., abs=1e-12)
    assert c.calc_theta() + p.calc_theta() == pytest.approx(c.model.risk_free_rate*c.model.calc_df(1.))
    assert c.calc_rho() + p.calc_rho() == pytest.approx(-c.model.calc_df(1.))


def test_time_and_zero_volatility():
    p = make_pricer()
    p.valuation_time = p.contract.expiry
    assert p.calc_fair_value() == 0.
    with pytest.raises(ValueError):
        p.calc_delta()
    p.valuation_time = 0.
    p.model.vol = 0.
    assert p.calc_fair_value() == pytest.approx(p.model.calc_df(1.))
    p.valuation_time = 2.
    with pytest.raises(ValueError):
        p.calc_fair_value()


def test_missing_spot_and_invalid_contract():
    p = make_pricer()
    with pytest.raises(ValueError):
        p.contract.payoff({})
    with pytest.raises(ValueError):
        EuropeanDigitalContract(Stock.BLUECHIP_BANK, PutCallFwd.FWD, LongShort.LONG, 100., 1.)
    vanilla = EuropeanContract(Stock.BLUECHIP_BANK, PutCallFwd.CALL, LongShort.LONG, 100., 1.)
    with pytest.raises(TypeError):
        EuropeanDigitalAnalyticPricer(vanilla, p.model, Params())
