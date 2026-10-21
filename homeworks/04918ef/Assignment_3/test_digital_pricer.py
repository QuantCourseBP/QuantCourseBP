import sys
from pathlib import Path

# Add current directory and QuantCourseBP root directory to sys.path
current = Path(__file__).resolve().parent
sys.path.append(str(current))
sys.path.append(str(current.parents[2]))

import numpy as np
import pytest
from pricer import *


@pytest.mark.parametrize('underlying', [Stock.BLUECHIP_BANK, Stock.TIPTOP_SOLUTIONS])
@pytest.mark.parametrize('strike_over_spot', np.arange(0.5, 2.0, 0.5))
@pytest.mark.parametrize('expiry', np.arange(0.5, 2.5, 0.5))
class TestDigitalParityAndFairValue:
    def test_digital_call_put_sum(self, underlying, strike_over_spot, expiry):
        params = Params()
        model = MarketModel(underlying)
        strike = strike_over_spot * model.spot

        call_contract = EuropeanDigitalContract(underlying, PutCallFwd.CALL, LongShort.LONG, strike, expiry)
        put_contract = EuropeanDigitalContract(underlying, PutCallFwd.PUT, LongShort.LONG, strike, expiry)

        call_pricer = EuropeanDigitalAnalyticPricer(call_contract, model, params)
        put_pricer = EuropeanDigitalAnalyticPricer(put_contract, model, params)

        call_fv = call_pricer.calc_fair_value()
        put_fv = put_pricer.calc_fair_value()
        df = model.calc_df(expiry)

        # Digital Call + Digital Put fair values must sum up to the discount factor (df)
        assert call_fv + put_fv == pytest.approx(df, abs=1e-6)

    def test_fair_value_bounds(self, underlying, strike_over_spot, expiry):
        params = Params()
        model = MarketModel(underlying)
        strike = strike_over_spot * model.spot

        call_contract = EuropeanDigitalContract(underlying, PutCallFwd.CALL, LongShort.LONG, strike, expiry)
        call_pricer = EuropeanDigitalAnalyticPricer(call_contract, model, params)
        fv = call_pricer.calc_fair_value()
        df = model.calc_df(expiry)

        # Fair value of long digital call must stay bounded between 0 and df
        assert 0.0 <= fv <= df