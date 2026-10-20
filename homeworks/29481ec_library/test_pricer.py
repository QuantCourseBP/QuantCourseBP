import numpy as np
import pytest
from pricer import *


# TASK:
# 1. Create put-call parity test for fair value and all greeks using various underlying, strike, expiry.


@pytest.mark.parametrize('underlying', [Stock.BLUECHIP_BANK, Stock.TIPTOP_SOLUTIONS])
@pytest.mark.parametrize('strike_over_spot', np.arange(0.5, 2.0, 0.5))
@pytest.mark.parametrize('expiry', np.arange(0.5, 2.5, 0.5))
class TestPutCallParity:
    def test_analytic_pricer(self, underlying, strike_over_spot, expiry):
        pricer = dict()
        result = dict()
        params = Params()
        model = MarketModel(underlying)
        strike = strike_over_spot * model.spot
        contract = ForwardContract(underlying, LongShort.LONG, strike, expiry)
        pricer['fwd'] = ForwardAnalyticPricer(contract, model, params)
        contract = EuropeanContract(underlying, PutCallFwd.CALL, LongShort.LONG, strike, expiry)
        pricer['call'] = EuropeanAnalyticPricer(contract, model, params)
        contract = EuropeanContract(underlying, PutCallFwd.PUT, LongShort.SHORT, strike, expiry)
        pricer['put'] = EuropeanAnalyticPricer(contract, model, params)
        for deriv_type in pricer.keys():
            result[deriv_type] = dict()
            result[deriv_type]['fv'] = pricer[deriv_type].calc_fair_value()
            result[deriv_type]['delta'] = pricer[deriv_type].calc_delta()
            result[deriv_type]['gamma'] = pricer[deriv_type].calc_gamma()
            result[deriv_type]['vega'] = pricer[deriv_type].calc_vega()
            result[deriv_type]['theta'] = pricer[deriv_type].calc_theta()
            result[deriv_type]['rho'] = pricer[deriv_type].calc_rho()
        result_put_call = {key: result['call'].get(key, 0) + result['put'].get(key, 0)
                           for key in set(result['call']) | set(result['put'])}
        assert result_put_call == pytest.approx(result['fwd'], abs=1e-6)


# ---------------------------------------------------------------------------
# Tests for the European digital option (EuropeanDigitalContract / EuropeanDigitalAnalyticPricer)
# ---------------------------------------------------------------------------

GREEK_METHODS = {
    'fv': 'calc_fair_value',
    'delta': 'calc_delta',
    'gamma': 'calc_gamma',
    'vega': 'calc_vega',
    'theta': 'calc_theta',
    'rho': 'calc_rho',
}


def make_digital_pricer(underlying, deriv_type, long_short, strike, expiry, payout=1.0):
    model = MarketModel(underlying)
    contract = EuropeanDigitalContract(underlying, deriv_type, long_short, strike, expiry, payout)
    return EuropeanDigitalAnalyticPricer(contract, model, Params())


@pytest.mark.parametrize('underlying', [Stock.BLUECHIP_BANK, Stock.TIPTOP_SOLUTIONS])
@pytest.mark.parametrize('strike_over_spot', np.arange(0.5, 2.0, 0.5))
@pytest.mark.parametrize('expiry', np.arange(0.5, 2.5, 0.5))
class TestDigitalPutCallParity:
    """Digital call + digital put (same strike, expiry) = zero coupon bond paying 1 at expiry."""

    def test_digital_parity(self, underlying, strike_over_spot, expiry):
        model = MarketModel(underlying)
        strike = strike_over_spot * model.spot
        call = make_digital_pricer(underlying, PutCallFwd.CALL, LongShort.LONG, strike, expiry)
        put = make_digital_pricer(underlying, PutCallFwd.PUT, LongShort.LONG, strike, expiry)
        df = model.calc_df(expiry)
        rate = model.risk_free_rate
        bond = {'fv': df, 'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': rate * df, 'rho': -expiry * df}
        total = {key: getattr(call, method)() + getattr(put, method)() for key, method in GREEK_METHODS.items()}
        assert total == pytest.approx(bond, abs=1e-8)


@pytest.mark.parametrize('deriv_type', [PutCallFwd.CALL, PutCallFwd.PUT])
@pytest.mark.parametrize('long_short', [LongShort.LONG, LongShort.SHORT])
@pytest.mark.parametrize('strike_over_spot', [0.8, 1.0, 1.2])
@pytest.mark.parametrize('expiry', [0.5, 1.0, 2.0])
class TestDigitalGreeksBumpAndReval:
    """Analytic greeks must match finite differences of the fair value."""

    def test_greeks(self, deriv_type, long_short, strike_over_spot, expiry):
        underlying = Stock.BLUECHIP_BANK
        spot = MarketModel(underlying).spot
        strike = strike_over_spot * spot
        pricer = make_digital_pricer(underlying, deriv_type, long_short, strike, expiry, payout=2.0)
        model = pricer.model
        base = {key: getattr(pricer, method)() for key, method in GREEK_METHODS.items()}

        def fair_value_with(**bumps):
            p = make_digital_pricer(underlying, deriv_type, long_short, strike, expiry, payout=2.0)
            p.model.spot += bumps.get('spot', 0.0)
            p.model.vol += bumps.get('vol', 0.0)
            p.model.risk_free_rate += bumps.get('rate', 0.0)
            p.valuation_time = bumps.get('time', 0.0)
            return p.calc_fair_value()

        h_spot, h_vol, h_rate, h_time = 1e-3 * spot, 1e-4, 1e-5, 1e-5
        up, down = fair_value_with(spot=h_spot), fair_value_with(spot=-h_spot)
        fd = {
            'delta': (up - down) / (2 * h_spot),
            'gamma': (up - 2 * base['fv'] + down) / h_spot ** 2,
            'vega': (fair_value_with(vol=h_vol) - fair_value_with(vol=-h_vol)) / (2 * h_vol),
            'rho': (fair_value_with(rate=h_rate) - fair_value_with(rate=-h_rate)) / (2 * h_rate),
            'theta': (fair_value_with(time=h_time) - fair_value_with(time=-h_time)) / (2 * h_time),
        }
        for key, value in fd.items():
            assert base[key] == pytest.approx(value, rel=1e-3, abs=1e-6), key


def test_digital_payoff():
    contract_call = EuropeanDigitalContract(Stock.BLUECHIP_BANK, PutCallFwd.CALL, LongShort.LONG, 100.0, 1.0)
    contract_put = EuropeanDigitalContract(Stock.BLUECHIP_BANK, PutCallFwd.PUT, LongShort.SHORT, 100.0, 1.0, payout=3.0)
    assert contract_call.payoff({1.0: 101.0}) == 1.0
    assert contract_call.payoff({1.0: 99.0}) == 0.0
    assert contract_put.payoff({1.0: 99.0}) == -3.0
    assert contract_put.payoff({1.0: 101.0}) == 0.0
    with pytest.raises(ValueError):
        EuropeanDigitalContract(Stock.BLUECHIP_BANK, PutCallFwd.FWD, LongShort.LONG, 100.0, 1.0)


@pytest.mark.parametrize('underlying', [Stock.BLUECHIP_BANK, Stock.TIPTOP_SOLUTIONS])
@pytest.mark.parametrize('expiry', [0.5, 1.0])
def test_digital_call_is_limit_of_bull_spreads(underlying, expiry):
    """Bull spread (long call K-w/2, short call K+w/2) divided by w -> digital call as w -> 0."""
    model = MarketModel(underlying)
    strike = model.spot
    digital = make_digital_pricer(underlying, PutCallFwd.CALL, LongShort.LONG, strike, expiry).calc_fair_value()
    width = 1e-3 * strike
    long_call = EuropeanContract(underlying, PutCallFwd.CALL, LongShort.LONG, strike - width / 2, expiry)
    short_call = EuropeanContract(underlying, PutCallFwd.CALL, LongShort.SHORT, strike + width / 2, expiry)
    bull_spread = (EuropeanAnalyticPricer(long_call, model, Params()).calc_fair_value()
                   + EuropeanAnalyticPricer(short_call, model, Params()).calc_fair_value()) / width
    assert bull_spread == pytest.approx(digital, rel=1e-4)
