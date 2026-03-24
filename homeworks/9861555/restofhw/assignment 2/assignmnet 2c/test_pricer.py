import numpy as np
import pytest
from pricer import *


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

        result_put_call = {
            key: result['call'].get(key, 0) + result['put'].get(key, 0)
            for key in set(result['call']) | set(result['put'])
        }

        assert result_put_call == pytest.approx(result['fwd'], abs=1e-6)


@pytest.mark.parametrize('underlying', [Stock.BLUECHIP_BANK, Stock.TIPTOP_SOLUTIONS])
@pytest.mark.parametrize('strike_over_spot', np.arange(0.5, 2.0, 0.5))
@pytest.mark.parametrize('expiry', np.arange(0.5, 2.5, 0.5))
class TestEuropeanDigitalParity:
    def test_analytic_pricer_fair_value(self, underlying, strike_over_spot, expiry):
        pricer = dict()
        result = dict()
        params = Params()
        model = MarketModel(underlying)
        strike = strike_over_spot * model.spot

        contract = EuropeanDigitalContract(
            underlying, PutCallFwd.CALL, LongShort.LONG, strike, expiry )
        pricer['call'] = EuropeanDigitalAnalyticPricer(contract, model, params)

        contract = EuropeanDigitalContract(
            underlying, PutCallFwd.PUT, LongShort.LONG, strike, expiry)
        pricer['put'] = EuropeanDigitalAnalyticPricer(contract, model, params)

        for deriv_type in pricer.keys():
            result[deriv_type] = dict()
            result[deriv_type]['fv'] = pricer[deriv_type].calc_fair_value()

        df = model.calc_df(expiry)

        assert result['call']['fv'] + result['put']['fv'] == pytest.approx(df, abs=1e-6)