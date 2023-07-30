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
            (PutCallFwd.CALL, FlatVolModel): 19.787749603901013,
            (PutCallFwd.PUT, BSVolModel): 5.336361001984066,
            (PutCallFwd.PUT, FlatVolModel): 5.747304317317166
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
            (PutCallFwd.CALL, GreekMethod.ANALYTIC): 0.7400021016466151,
            (PutCallFwd.CALL, GreekMethod.BUMP): 0.7399483934813667,
            (PutCallFwd.PUT, GreekMethod.ANALYTIC): -0.25999789835338494,
            (PutCallFwd.PUT, GreekMethod.BUMP): -0.2600516065186316
        }
        contract = EuropeanContract(self.und, derivative_type, self.ls, self.strike, self.expiry)
        mod = FlatVolModel(self.und)
        method = AnalyticMethod(mod)
        pricer = EuropeanAnalyticPricer(contract, mod, method)
        delta = pricer.calc_delta(greek_method)
        assert delta == pytest.approx(expected_result[(derivative_type, greek_method)])
        assert expected_result[(derivative_type, GreekMethod.ANALYTIC)] == \
               pytest.approx(expected_result[(derivative_type, GreekMethod.BUMP)], rel=1e-2)


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

