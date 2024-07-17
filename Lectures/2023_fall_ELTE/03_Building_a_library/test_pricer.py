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

        # Create contracts and pricers here
        pricer['fwd'] = ...
        pricer['call'] = ...
        pricer['put'] = ...

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
