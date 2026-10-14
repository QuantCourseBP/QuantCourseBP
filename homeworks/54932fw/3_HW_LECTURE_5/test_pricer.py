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


# =====================================================================================================
# Homework 3, Q1 (NEW CODE): unit tests of the digital option.
# Fair value (asked in the assignment) + payoff, type checks and Greeks.
# =====================================================================================================
def digital_pricer(model: MarketModel, call_put: PutCallFwd, long_short: LongShort, strike: float,
                   expiry: float) -> EuropeanDigitalAnalyticPricer:
    contract = EuropeanDigitalContract(model.underlying, call_put, long_short, strike, expiry)
    return EuropeanDigitalAnalyticPricer(contract, model, Params())


def call_price(model: MarketModel, strike: float, expiry: float) -> float:
    """Long vanilla call of the lecture (EuropeanAnalyticPricer)."""
    contract = EuropeanContract(model.underlying, PutCallFwd.CALL, LongShort.LONG, strike, expiry)
    return EuropeanAnalyticPricer(contract, model, Params()).calc_fair_value()


@pytest.mark.parametrize('underlying', [Stock.BLUECHIP_BANK, Stock.TIPTOP_SOLUTIONS])
@pytest.mark.parametrize('strike_over_spot', np.arange(0.5, 2.0, 0.5))
@pytest.mark.parametrize('expiry', np.arange(0.5, 2.5, 0.5))
class TestDigitalFairValue:
    def test_digital_parity(self, underlying, strike_over_spot, expiry):
        # long digital call + long digital put pay 1 at T, worth DF = e^{-rT} today (Lemma 4)
        model = MarketModel(underlying)
        strike = strike_over_spot * model.spot
        call = digital_pricer(model, PutCallFwd.CALL, LongShort.LONG, strike, expiry).calc_fair_value()
        put = digital_pricer(model, PutCallFwd.PUT, LongShort.LONG, strike, expiry).calc_fair_value()
        assert call + put == pytest.approx(model.calc_df(expiry), abs=1e-12)

    def test_bull_spread_bounds(self, underlying, strike_over_spot, expiry):
        # (c(K) - c(K+h))/h <= C_d <= (c(K-h) - c(K))/h for every h > 0 (proof of Lemma 4); h = 1% of K
        model = MarketModel(underlying)
        strike = strike_over_spot * model.spot
        h = 0.01 * strike
        digital = digital_pricer(model, PutCallFwd.CALL, LongShort.LONG, strike, expiry).calc_fair_value()
        lower = (call_price(model, strike, expiry) - call_price(model, strike + h, expiry)) / h
        upper = (call_price(model, strike - h, expiry) - call_price(model, strike, expiry)) / h
        assert lower <= digital <= upper

    def test_bull_spread_limit(self, underlying, strike_over_spot, expiry):
        # a narrow scaled bull spread (width h = 0.01% of K) has the price of the digital call (Lemma 4)
        model = MarketModel(underlying)
        strike = strike_over_spot * model.spot
        h = 1e-4 * strike
        digital = digital_pricer(model, PutCallFwd.CALL, LongShort.LONG, strike, expiry).calc_fair_value()
        spread = (call_price(model, strike - h / 2, expiry) - call_price(model, strike + h / 2, expiry)) / h
        assert spread == pytest.approx(digital, abs=1e-6)

    def test_short_is_minus_long(self, underlying, strike_over_spot, expiry):
        model = MarketModel(underlying)
        strike = strike_over_spot * model.spot
        for call_put in (PutCallFwd.CALL, PutCallFwd.PUT):
            long = digital_pricer(model, call_put, LongShort.LONG, strike, expiry).calc_fair_value()
            short = digital_pricer(model, call_put, LongShort.SHORT, strike, expiry).calc_fair_value()
            assert short == pytest.approx(-long, abs=1e-15)


def test_digital_value_by_hand():
    # BLUECHIP_BANK (S = 130.17, vol = 0.381, r = 0.05), K = 130, T = 1:
    # d2 = -0.055836, DF = 0.951229, C_d = DF * N(d2) = 0.454437, P_d = DF * N(-d2) = 0.496793
    model = MarketModel(Stock.BLUECHIP_BANK)
    call = digital_pricer(model, PutCallFwd.CALL, LongShort.LONG, 130.0, 1.0).calc_fair_value()
    put = digital_pricer(model, PutCallFwd.PUT, LongShort.LONG, 130.0, 1.0).calc_fair_value()
    assert call == pytest.approx(0.454437, abs=1e-6)
    assert put == pytest.approx(0.496793, abs=1e-6)


def test_digital_payoff():
    long_call = EuropeanDigitalContract(Stock.BLUECHIP_BANK, PutCallFwd.CALL, LongShort.LONG, 130.0, 1.0)
    short_put = EuropeanDigitalContract(Stock.BLUECHIP_BANK, PutCallFwd.PUT, LongShort.SHORT, 130.0, 1.0)
    assert long_call.payoff({1.0: 131.0}) == 1.0
    assert long_call.payoff({1.0: 129.0}) == 0.0
    assert short_put.payoff({1.0: 129.0}) == -1.0
    assert short_put.payoff({1.0: 131.0}) == 0.0
    with pytest.raises(ValueError):
        long_call.payoff({0.5: 131.0})                 # no spot at the expiry


def test_digital_type_checks():
    model = MarketModel(Stock.BLUECHIP_BANK)
    vanilla = EuropeanContract(Stock.BLUECHIP_BANK, PutCallFwd.CALL, LongShort.LONG, 130.0, 1.0)
    digital = EuropeanDigitalContract(Stock.BLUECHIP_BANK, PutCallFwd.CALL, LongShort.LONG, 130.0, 1.0)
    with pytest.raises(ValueError):
        EuropeanDigitalContract(Stock.BLUECHIP_BANK, PutCallFwd.FWD, LongShort.LONG, 130.0, 1.0)
    with pytest.raises(TypeError):
        EuropeanDigitalAnalyticPricer(vanilla, model, Params())      # a vanilla is not a digital
    with pytest.raises(TypeError):
        EuropeanAnalyticPricer(digital, model, Params())             # the vanilla pricer rejects a digital


def bumped_value(pricer: EuropeanDigitalAnalyticPricer, attribute: str, h: float) -> float:
    """Fair value after adding h to the model's spot / vol / risk_free_rate, or to the valuation time."""
    owner = pricer if attribute == 'valuation_time' else pricer.model
    base = getattr(owner, attribute)
    setattr(owner, attribute, base + h)
    value = pricer.calc_fair_value()
    setattr(owner, attribute, base)
    return value


@pytest.mark.parametrize('call_put', [PutCallFwd.CALL, PutCallFwd.PUT])
@pytest.mark.parametrize('long_short', [LongShort.LONG, LongShort.SHORT])
def test_digital_greeks_bump(call_put, long_short):
    # analytic Greeks (Lemma 5) = central differences of the fair value
    pricer = digital_pricer(MarketModel(Stock.BLUECHIP_BANK), call_put, long_short, 130.0, 1.0)
    fv = pricer.calc_fair_value()
    hs, h = 0.01, 1e-4
    bump = {
        'delta': (bumped_value(pricer, 'spot', hs) - bumped_value(pricer, 'spot', -hs)) / (2 * hs),
        'gamma': (bumped_value(pricer, 'spot', hs) - 2 * fv + bumped_value(pricer, 'spot', -hs)) / hs**2,
        'vega': (bumped_value(pricer, 'vol', h) - bumped_value(pricer, 'vol', -h)) / (2 * h),
        'theta': (bumped_value(pricer, 'valuation_time', h) - bumped_value(pricer, 'valuation_time', -h)) / (2 * h),
        'rho': (bumped_value(pricer, 'risk_free_rate', h) - bumped_value(pricer, 'risk_free_rate', -h)) / (2 * h)}
    analytic = {'delta': pricer.calc_delta(), 'gamma': pricer.calc_gamma(), 'vega': pricer.calc_vega(),
                'theta': pricer.calc_theta(), 'rho': pricer.calc_rho()}
    assert analytic == pytest.approx(bump, rel=1e-4, abs=1e-8)
