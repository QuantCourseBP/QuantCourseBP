import numpy as np
import pytest
from src.pricer import *

MarketData.initialize()


class TestEuropeanAnalyticPricer:
    und = Stock.EXAMPLE1
    expiry = 2.0
    strike = 0.95 * MarketData.get_initial_spot()[und]
    ls = LongShort.LONG

    @pytest.mark.parametrize('derivative_type', [PutCallFwd.CALL, PutCallFwd.PUT])
    @pytest.mark.parametrize('model', [BSVolModel, FlatVolModel])
    def test_fair_value(self, derivative_type, model):
        expected_result = {
            (PutCallFwd.CALL, BSVolModel): 19.376806288567913,
            (PutCallFwd.CALL, FlatVolModel): 18.790736019031833,
            (PutCallFwd.PUT, BSVolModel): 5.336361001984066,
            (PutCallFwd.PUT, FlatVolModel): 4.750290732447986
        }
        contract = EuropeanContract(self.und, derivative_type, self.ls, self.strike, self.expiry)
        mod = model(self.und)
        method = AnalyticMethod(mod)
        pricer = EuropeanAnalyticPricer(contract, mod, method)
        fv = pricer.calc_fair_value()
        assert fv == pytest.approx(expected_result[(derivative_type, model)])

    @pytest.mark.parametrize('derivative_type', [PutCallFwd.CALL, PutCallFwd.PUT])
    @pytest.mark.parametrize('greek_method', [GreekMethod.ANALYTIC, GreekMethod.BUMP])
    def test_delta(self, derivative_type, greek_method):
        expected_result = {
            (PutCallFwd.CALL, GreekMethod.ANALYTIC): 0.7524906409516483,
            (PutCallFwd.CALL, GreekMethod.BUMP): 0.7524254274946358,
            (PutCallFwd.PUT, GreekMethod.ANALYTIC): -0.24750935904835175,
            (PutCallFwd.PUT, GreekMethod.BUMP): -0.24757457250536063
        }
        contract = EuropeanContract(self.und, derivative_type, self.ls, self.strike, self.expiry)
        mod = FlatVolModel(self.und)
        method = AnalyticMethod(mod)
        pricer = EuropeanAnalyticPricer(contract, mod, method)
        delta = pricer.calc_delta(greek_method)
        assert delta == pytest.approx(expected_result[(derivative_type, greek_method)])
        assert expected_result[(derivative_type, GreekMethod.ANALYTIC)] == \
               pytest.approx(expected_result[(derivative_type, GreekMethod.BUMP)], rel=1e-3)


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
    und = Stock.EXAMPLE1
    model = FlatVolModel(und)
    expiry = 1.0
    strike = 1.0 * MarketData.get_initial_spot()[und]
    contract = EuropeanContract(und, PutCallFwd.CALL, LongShort.LONG, strike, expiry)
    
    params = [TreeParams(expiry, strike, 2, 1.2, 0.8), TreeParams(expiry, strike, 2)]
    pvs = [7.982310163583209, 10.473492643687308]

    @pytest.mark.parametrize('param, expected_pv', zip(params, pvs))
    def test_tree_pricer(self, param, expected_pv):
        pricer = GenericTreePricer(TestTreePricer.contract, TestTreePricer.model, param)
        pv = pricer.calc_fair_value()
        assert pv == pytest.approx(expected_pv)

