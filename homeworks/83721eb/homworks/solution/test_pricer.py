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
def test_european_digital_call_pricer():
    params = Params()
    model = MarketModel(Stock.BLUECHIP_BANK)

    strike = model.spot
    expiry = 1.0

    contract = EuropeanDigitalContract(
        Stock.BLUECHIP_BANK,
        PutCallFwd.CALL,
        LongShort.LONG,
        strike,
        expiry
    )

    pricer = EuropeanDigitalAnalyticPricer(contract, model, params)

    value = pricer.calc_fair_value()

    assert value >= 0.0
    assert value <= 1.0

def test_european_digital_greeks_run():
        params = Params()
        model = MarketModel(Stock.BLUECHIP_BANK)

        strike = model.spot
        expiry = 1.0

        contract = EuropeanDigitalContract(
            Stock.BLUECHIP_BANK,
            PutCallFwd.CALL,
            LongShort.LONG,
            strike,
            expiry
        )

        pricer = EuropeanDigitalAnalyticPricer(contract, model, params)

        values = [
            pricer.calc_fair_value(),
            pricer.calc_delta(),
            pricer.calc_gamma(),
            pricer.calc_vega(),
            pricer.calc_theta(),
            pricer.calc_rho(),
        ]

        for value in values:
            assert np.isfinite(value)