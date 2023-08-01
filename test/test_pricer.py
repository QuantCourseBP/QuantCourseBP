import numpy as np
import pytest
from src.pricer import *


@pytest.mark.parametrize('ref_strike', [0.9, 1.0])
@pytest.mark.parametrize('expiry', [0.5, 2.0])
class TestForwardAnalyticPricer:
    MarketData.initialize()
    und = Stock.EXAMPLE1
    ls = LongShort.LONG
    model = FlatVolModel(und)
    method = AnalyticMethod(model)

    def test_fair_value(self, ref_strike, expiry):
        expected_result = {
            (round(0.9, 1), round(0.5, 1)): 12.22210791745006,
            (round(0.9, 1), round(2.0, 1)): 18.56463237676364,
            (round(1.0, 1), round(0.5, 1)): 2.4690087971667367,
            (round(1.0, 1), round(2.0, 1)): 9.516258196404053
        }
        strike = ref_strike * MarketData.get_initial_spot()[self.und]
        contract = ForwardContract(self.und, self.ls, strike, expiry)
        pricer = ForwardAnalyticPricer(contract, self.model, self.method)
        fv = pricer.calc_fair_value()
        assert fv == pytest.approx(expected_result[(round(ref_strike, 1), round(expiry, 1))])

    @pytest.mark.parametrize('greek_method', [GreekMethod.ANALYTIC, GreekMethod.BUMP])
    def test_delta(self, ref_strike, expiry, greek_method):
        expected_result = {
            GreekMethod.ANALYTIC: 1.0,
            GreekMethod.BUMP: 1.0
        }
        strike = ref_strike * MarketData.get_initial_spot()[self.und]
        contract = ForwardContract(self.und, self.ls, strike, expiry)
        pricer = ForwardAnalyticPricer(contract, self.model, self.method)
        delta = pricer.calc_delta(greek_method)
        assert delta == pytest.approx(expected_result[greek_method])
        assert expected_result[GreekMethod.ANALYTIC] == pytest.approx(expected_result[GreekMethod.BUMP], rel=1e-3)


class TestTreePricer:
    MarketData.initialize()
    
    model = FlatVolModel(Stock.EXAMPLE1)
    expiry = 1.0
    moneyness = 1.0
    contract = EuropeanContract(Stock.EXAMPLE1, PutCallFwd.CALL, LongShort.LONG, moneyness, expiry)
    
    params = [TreeParams(expiry, moneyness, 2, 1.2, 0.8), TreeParams(expiry, moneyness, 2)]
    pvs = [0.0798231016, 0.104734926]

    @pytest.mark.parametrize('param, expected_pv', zip(params, pvs))
    def test_tree_pricer(self, param, expected_pv):
        pricer = GenericTreePricer(TestTreePricer.contract, TestTreePricer.model, param)
        pv = pricer.calc_fair_value()
        spot_tree = pricer._tree_method._spot_tree
        assert pv == pytest.approx(expected_pv)

