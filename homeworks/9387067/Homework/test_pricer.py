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
       
    
@pytest.mark.parametrize('underlying', [Stock.BLUECHIP_BANK, Stock.TIPTOP_SOLUTIONS])
@pytest.mark.parametrize('strike_over_spot', np.arange(0.5, 2.0, 0.5))
@pytest.mark.parametrize('expiry', np.arange(0.5, 2.5, 0.5))
@pytest.mark.parametrize('cash_payoff', np.arange(1, 6, 1))
class TestFairValue:
    def test_digital_analytic_pricer(self, underlying, strike_over_spot, expiry, cash_payoff):
        pricer = dict()
        result = dict()
        expected = dict()
        params = Params()
        model = MarketModel(underlying)
        strike = strike_over_spot * model.spot
        spot_over_strike = model.spot / strike
        df = model.calc_df(expiry)
        rate = model.risk_free_rate
        vol = model.vol
        d1 = 1 / (vol * np.sqrt(expiry)) * (np.log(spot_over_strike) + (rate + vol**2 / 2) * expiry)
        d2 = d1 - vol * np.sqrt(expiry)
        contract = EuropeanDigitalContract(underlying, PutCallFwd.CALL, LongShort.LONG, strike, expiry, cash_payoff)
        pricer['long_call'] = EuropeanDigitalAnalyticPricer(contract, model, params)
        expected['long_call'] = cash_payoff * df * norm.cdf(d2)
        contract = EuropeanDigitalContract(underlying, PutCallFwd.CALL, LongShort.SHORT, strike, expiry, cash_payoff)
        pricer['short_call'] = EuropeanDigitalAnalyticPricer(contract, model, params)
        expected['short_call'] = -1.0 * cash_payoff * df * norm.cdf(d2)
        contract = EuropeanDigitalContract(underlying, PutCallFwd.PUT, LongShort.LONG, strike, expiry, cash_payoff)
        pricer['long_put'] = EuropeanDigitalAnalyticPricer(contract, model, params)
        expected['long_put'] = cash_payoff * df * norm.cdf(-d2)
        contract = EuropeanDigitalContract(underlying, PutCallFwd.PUT, LongShort.SHORT, strike, expiry, cash_payoff)
        pricer['short_put'] = EuropeanDigitalAnalyticPricer(contract, model, params)
        expected['short_put'] = -1.0 * cash_payoff * df * norm.cdf(-d2)
        for deriv_type in pricer.keys():
            result[deriv_type] = pricer[deriv_type].calc_fair_value()
        assert result == pytest.approx(expected, abs=1e-6)
        