"""Independent fair-value, Greek, contract and bull-spread verification."""
from copy import deepcopy
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm
from .contract import EuropeanContract, EuropeanDigitalContract
from .model import MarketModel
from .pricer import EuropeanAnalyticPricer, EuropeanDigitalAnalyticPricer, Params
from src.enums import Stock, PutCallFwd, LongShort


def make_pricer(option_type=PutCallFwd.CALL, position=LongShort.LONG,
                spot=100., strike=100., expiry=1., rate=.05, vol=.2):
    model = MarketModel(Stock.BLUECHIP_BANK)
    model.spot, model.vol, model.risk_free_rate = spot, vol, rate
    contract = EuropeanDigitalContract(Stock.BLUECHIP_BANK, option_type, position, strike, expiry)
    return EuropeanDigitalAnalyticPricer(contract, model, Params())


@pytest.mark.parametrize('option_type,expected', [(PutCallFwd.CALL, .5323248154537634),
                                                 (PutCallFwd.PUT, .41890460904695065)])
@pytest.mark.parametrize('position', [LongShort.LONG, LongShort.SHORT])
def test_reference_fair_value(option_type, expected, position):
    pricer = make_pricer(option_type, position)
    assert pricer.calc_fair_value() == pytest.approx(pricer.contract.direction*expected, abs=1e-12)


@pytest.mark.parametrize('option_type', [PutCallFwd.CALL, PutCallFwd.PUT])
@pytest.mark.parametrize('position', [LongShort.LONG, LongShort.SHORT])
@pytest.mark.parametrize('strike', [80., 100., 120.])
@pytest.mark.parametrize('expiry', [.25, 2.])
@pytest.mark.parametrize('rate', [-.02, .05])
def test_fair_value_against_integrated_terminal_payoff(option_type, position, strike, expiry, rate):
    pricer = make_pricer(option_type, position, strike=strike, expiry=expiry, rate=rate, vol=.3)
    # Under risk-neutral BSM: log(S_T/S0) = (r-vol^2/2)*T + vol*sqrt(T)*Z.
    # Integrate the terminal indicator against the normal density instead of
    # reusing the implementation's d2/cdf expression.
    threshold = (np.log(strike/100.)-(rate-.5*.3**2)*expiry)/(.3*np.sqrt(expiry))
    bounds = (threshold, np.inf) if option_type == PutCallFwd.CALL else (-np.inf, threshold)
    probability = quad(norm.pdf, *bounds, epsabs=1e-12)[0]
    expected = pricer.contract.direction*np.exp(-rate*expiry)*probability
    assert pricer.calc_fair_value() == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize('option_type', [PutCallFwd.CALL, PutCallFwd.PUT])
@pytest.mark.parametrize('position', [LongShort.LONG, LongShort.SHORT])
@pytest.mark.parametrize('spot', [80., 100., 120.])
@pytest.mark.parametrize('rate', [-.02, .05])
def test_all_greeks_against_finite_differences(option_type, position, spot, rate):
    pricer = make_pricer(option_type, position, spot=spot, rate=rate, expiry=1.7, vol=.3)
    pricer.valuation_time = .2  # Verify remaining time rather than full expiry.
    def value_after(field, step):
        bumped = deepcopy(pricer)
        if field == 'valuation_time':
            bumped.valuation_time += step
        else:
            setattr(bumped.model, field, getattr(bumped.model, field)+step)
        return bumped.calc_fair_value()
    for greek, field, h in [('delta', 'spot', .001), ('vega', 'vol', 1e-5),
                             ('theta', 'valuation_time', 1e-5), ('rho', 'risk_free_rate', 1e-5)]:
        expected = (value_after(field, h)-value_after(field, -h))/(2*h)
        assert getattr(pricer, 'calc_'+greek)() == pytest.approx(expected, rel=2e-6, abs=1e-9)
    h = .01
    gamma = (value_after('spot', h)-2*pricer.calc_fair_value()+value_after('spot', -h))/h**2
    assert pricer.calc_gamma() == pytest.approx(gamma, rel=2e-5, abs=1e-9)


@pytest.mark.parametrize('rate', [-.02, 0., .05])
def test_digital_call_plus_put_is_discount_bond_for_all_measures(rate):
    call = make_pricer(rate=rate, expiry=2.)
    put = make_pricer(PutCallFwd.PUT, rate=rate, expiry=2.)
    call.valuation_time = put.valuation_time = .5
    tau = 1.5
    df = np.exp(-rate*tau)
    expected = dict(fair_value=df, delta=0., gamma=0., vega=0., theta=rate*df, rho=-tau*df)
    for name, result in expected.items():
        total = getattr(call, 'calc_'+name)()+getattr(put, 'calc_'+name)()
        assert total == pytest.approx(result, abs=1e-12)


@pytest.mark.parametrize('option_type', [PutCallFwd.CALL, PutCallFwd.PUT])
@pytest.mark.parametrize('position', [LongShort.LONG, LongShort.SHORT])
@pytest.mark.parametrize('terminal', [90., 100., 110.])
def test_digital_payoff(option_type, position, terminal):
    contract = EuropeanDigitalContract(Stock.BLUECHIP_BANK, option_type, position, 100., 1.12345678)
    timeline = contract.get_timeline()
    assert timeline == [1.123457]
    indicator = terminal > 100 if option_type == PutCallFwd.CALL else terminal < 100
    assert contract.payoff({timeline[0]: terminal}) == contract.direction*float(indicator)


def test_missing_expiry_observation():
    with pytest.raises(ValueError, match='expects spot price on timeline'):
        make_pricer().contract.payoff({.5: 110.})


def test_invalid_derivative_and_pricer_contract_types():
    with pytest.raises(ValueError, match='must be one of'):
        EuropeanDigitalContract(Stock.BLUECHIP_BANK, PutCallFwd.FWD, LongShort.LONG, 100., 1.)
    vanilla = EuropeanContract(Stock.BLUECHIP_BANK, PutCallFwd.CALL, LongShort.LONG, 100., 1.)
    with pytest.raises(TypeError):
        EuropeanDigitalAnalyticPricer(vanilla, MarketModel(Stock.BLUECHIP_BANK), Params())
    with pytest.raises(TypeError):
        EuropeanAnalyticPricer(make_pricer().contract, MarketModel(Stock.BLUECHIP_BANK), Params())


@pytest.mark.parametrize('field,value', [('spot', 0.), ('vol', 0.), ('risk_free_rate', np.nan)])
def test_invalid_market_inputs(field, value):
    pricer = make_pricer()
    setattr(pricer.model, field, value)
    with pytest.raises(ValueError):
        pricer.calc_fair_value()


@pytest.mark.parametrize('valuation_time', [1., 2.])
def test_nonpositive_remaining_time(valuation_time):
    pricer = make_pricer()
    pricer.valuation_time = valuation_time
    with pytest.raises(ValueError, match='remaining time'):
        pricer.calc_fair_value()


@pytest.mark.parametrize('strike,expiry', [(0., 1.), (100., 0.), (np.inf, 1.)])
def test_invalid_contract_inputs(strike, expiry):
    with pytest.raises(ValueError):
        make_pricer(strike=strike, expiry=expiry)


@pytest.mark.parametrize('spot', [80., 100., 120.])
def test_narrow_bull_spreads_converge_to_digital_price(spot):
    digital = make_pricer(spot=spot)
    K = digital.contract.strike
    def vanilla(strike, position):
        contract = EuropeanContract(Stock.BLUECHIP_BANK, PutCallFwd.CALL, position, strike, 1.)
        return EuropeanAnalyticPricer(contract, digital.model, Params()).calc_fair_value()
    errors = []
    for width in [4., 1., .25, .0625]:
        # Long K call, short K+width call, 1/width units of each.
        spread = (vanilla(K, LongShort.LONG)+vanilla(K+width, LongShort.SHORT))/width
        assert 0 <= spread <= digital.model.calc_df(1.)
        errors.append(abs(spread-digital.calc_fair_value()))
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] < .001
