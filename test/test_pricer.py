import numpy as np
import pytest
from src.pricer import *

MarketData.initialize()


@pytest.mark.parametrize('strike', [0.9, 1.0])
@pytest.mark.parametrize('expiry', [0.5, 2.0])
class TestForwardAnalyticPricer:
    MarketData.initialize()
    und = Stock.EXAMPLE1
    ls = LongShort.LONG
    model = FlatVolModel(und)
    method = AnalyticMethod(model)

    def test_fair_value(self, strike, expiry):
        expected_result = {
            (round(0.9, 1), round(0.5, 1)): 12.22210791745006,
            (round(0.9, 1), round(2.0, 1)): 18.56463237676364,
            (round(1.0, 1), round(0.5, 1)): 2.4690087971667367,
            (round(1.0, 1), round(2.0, 1)): 9.516258196404053
        }
        strike_level = strike * MarketData.get_initial_spot()[self.und]
        contract = ForwardContract(self.und, self.ls, strike_level, expiry)
        pricer = ForwardAnalyticPricer(contract, self.model, self.method)
        fv = pricer.calc_fair_value()
        assert fv == pytest.approx(expected_result[(round(strike, 1), round(expiry, 1))])

    @pytest.mark.parametrize('greek_method', [GreekMethod.ANALYTIC, GreekMethod.BUMP])
    def test_delta(self, strike, expiry, greek_method):
        expected_result = {
            GreekMethod.ANALYTIC: 1.0,
            GreekMethod.BUMP: 1.0
        }
        strike = strike * MarketData.get_initial_spot()[self.und]
        contract = ForwardContract(self.und, self.ls, strike, expiry)
        pricer = ForwardAnalyticPricer(contract, self.model, self.method)
        delta = pricer.calc_delta(greek_method)
        assert delta == pytest.approx(expected_result[greek_method])
        assert expected_result[GreekMethod.ANALYTIC] == pytest.approx(expected_result[GreekMethod.BUMP], rel=1e-3)


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


@pytest.mark.parametrize('underlying', [Stock.EXAMPLE1, Stock.EXAMPLE2])
@pytest.mark.parametrize('ref_strike', np.arange(0.5, 2.0, 0.5))
@pytest.mark.parametrize('expiry', np.arange(0.5, 2.5, 0.5))
@pytest.mark.parametrize('model', [BSVolModel, FlatVolModel])
class TestPutCallParity:
    @pytest.mark.parametrize('greek_method', [GreekMethod.ANALYTIC, GreekMethod.BUMP])
    def test_analytic_pricer(self, underlying, ref_strike, expiry, model, greek_method):
        pricer = dict()
        result = dict()
        mod = model(underlying)
        strike = ref_strike * MarketData.get_initial_spot()[underlying]
        contract = ForwardContract(underlying, LongShort.LONG, strike, expiry)
        pricer['fwd'] = ForwardAnalyticPricer(contract, mod, AnalyticMethod(mod))
        contract = EuropeanContract(underlying, PutCallFwd.CALL, LongShort.LONG, strike, expiry)
        pricer['call'] = EuropeanAnalyticPricer(contract, mod, AnalyticMethod(mod))
        contract = EuropeanContract(underlying, PutCallFwd.PUT, LongShort.SHORT, strike, expiry)
        pricer['put'] = EuropeanAnalyticPricer(contract, mod, AnalyticMethod(mod))
        for deriv_type in pricer.keys():
            result[deriv_type] = dict()
            result[deriv_type]['fv'] = pricer[deriv_type].calc_fair_value()
            result[deriv_type]['delta'] = pricer[deriv_type].calc_delta(greek_method)
            result[deriv_type]['gamma'] = pricer[deriv_type].calc_gamma(greek_method)
            result[deriv_type]['vega'] = pricer[deriv_type].calc_vega(greek_method)
            result[deriv_type]['theta'] = pricer[deriv_type].calc_theta(greek_method)
            result[deriv_type]['rho'] = pricer[deriv_type].calc_rho(greek_method)
        result_put_call = {key: result['call'].get(key, 0) + result['put'].get(key, 0)
                           for key in set(result['call']) | set(result['put'])}
        assert result_put_call == pytest.approx(result['fwd'], abs=1e-6)
