import numpy as np
import pytest
from src.pricer import *

MarketData.initialize()


@pytest.mark.parametrize('strike', [0.9, 1.0])
@pytest.mark.parametrize('expiry', [0.5, 2.0])
class TestForwardAnalyticPricer:
    und = Stock.TEST_COMPANY
    ls = LongShort.LONG
    model = FlatVolModel(und)

    def test_fair_value(self, strike, expiry):
        expected_result = {
            (round(0.9, 1), round(0.5, 1)): 12.22210791745006,
            (round(0.9, 1), round(2.0, 1)): 18.56463237676364,
            (round(1.0, 1), round(0.5, 1)): 2.4690087971667367,
            (round(1.0, 1), round(2.0, 1)): 9.516258196404053
        }
        strike_level = strike * MarketData.get_spot()[self.und]
        contract = ForwardContract(self.und, self.ls, strike_level, expiry)
        pricer = ForwardAnalyticPricer(contract, self.model, Params())
        fv = pricer.calc_fair_value()
        assert fv == pytest.approx(expected_result[(round(strike, 1), round(expiry, 1))])

    @pytest.mark.parametrize('greek_method', [GreekMethod.ANALYTIC, GreekMethod.BUMP])
    def test_delta(self, strike, expiry, greek_method):
        expected_result = {
            GreekMethod.ANALYTIC: 1.0,
            GreekMethod.BUMP: 1.0
        }
        strike = strike * MarketData.get_spot()[self.und]
        contract = ForwardContract(self.und, self.ls, strike, expiry)
        pricer = ForwardAnalyticPricer(contract, self.model, Params())
        delta = pricer.calc_delta(greek_method)
        assert delta == pytest.approx(expected_result[greek_method])
        assert expected_result[GreekMethod.ANALYTIC] == pytest.approx(expected_result[GreekMethod.BUMP], rel=1e-3)


class TestEuropeanAnalyticPricer:
    und = Stock.TEST_COMPANY
    expiry = 2.0
    strike = 0.95 * MarketData.get_spot()[und]
    ls = LongShort.LONG

    @pytest.mark.parametrize('derivative_type', [PutCallFwd.CALL, PutCallFwd.PUT])
    @pytest.mark.parametrize('model', [BSVolModel, FlatVolModel])
    def test_fair_value(self, derivative_type, model):
        expected_result = {
            (PutCallFwd.CALL, BSVolModel): 23.60627624868423,
            (PutCallFwd.CALL, FlatVolModel): 19.558965822125977,
            (PutCallFwd.PUT, BSVolModel): 9.565830962100371,
            (PutCallFwd.PUT, FlatVolModel): 5.518520535542123
        }
        contract = EuropeanContract(self.und, derivative_type, self.ls, self.strike, self.expiry)
        mod = model(self.und)
        pricer = EuropeanAnalyticPricer(contract, mod, Params())
        fv = pricer.calc_fair_value()
        assert fv == pytest.approx(expected_result[(derivative_type, model)])

    @pytest.mark.parametrize('derivative_type', [PutCallFwd.CALL, PutCallFwd.PUT])
    @pytest.mark.parametrize('greek_method', [GreekMethod.ANALYTIC, GreekMethod.BUMP])
    def test_delta(self, derivative_type, greek_method):
        expected_result = {
            (PutCallFwd.CALL, GreekMethod.ANALYTIC): 0.7425509208287824,
            (PutCallFwd.CALL, GreekMethod.BUMP): 0.7424949126182163,
            (PutCallFwd.PUT, GreekMethod.ANALYTIC): -0.2574490791712177,
            (PutCallFwd.PUT, GreekMethod.BUMP): -0.2575050873817766
        }
        contract = EuropeanContract(self.und, derivative_type, self.ls, self.strike, self.expiry)
        mod = FlatVolModel(self.und)
        pricer = EuropeanAnalyticPricer(contract, mod, Params())
        greek = pricer.calc_delta(greek_method)
        assert greek == pytest.approx(expected_result[(derivative_type, greek_method)])
        assert expected_result[(derivative_type, GreekMethod.ANALYTIC)] == \
               pytest.approx(expected_result[(derivative_type, GreekMethod.BUMP)], rel=1e-2)

    @pytest.mark.parametrize('derivative_type', [PutCallFwd.CALL, PutCallFwd.PUT])
    @pytest.mark.parametrize('greek_method', [GreekMethod.ANALYTIC, GreekMethod.BUMP])
    def test_theta(self, derivative_type, greek_method):
        expected_result = {
            (PutCallFwd.CALL, GreekMethod.ANALYTIC): -5.1764745118252495,
            (PutCallFwd.CALL, GreekMethod.BUMP): -5.177663427889634,
            (PutCallFwd.PUT, GreekMethod.ANALYTIC): -0.878496776154442,
            (PutCallFwd.PUT, GreekMethod.BUMP): -0.8793912967345996
        }
        contract = EuropeanContract(self.und, derivative_type, self.ls, self.strike, self.expiry)
        mod = FlatVolModel(self.und)
        pricer = EuropeanAnalyticPricer(contract, mod, Params())
        greek = pricer.calc_theta(greek_method)
        assert greek == pytest.approx(expected_result[(derivative_type, greek_method)])
        assert expected_result[(derivative_type, GreekMethod.ANALYTIC)] == \
               pytest.approx(expected_result[(derivative_type, GreekMethod.BUMP)], rel=1e-2)


class TestTreePricer:
    und = Stock.TEST_COMPANY
    model = FlatVolModel(und)
    expiry = 1.0
    strike = 1.0 * MarketData.get_spot()[und]

    europeanParams = [TreeParams(2, np.nan, 1.2, 0.8), TreeParams(2, 0.2),
                      TreeParams(2), TreeParams(2), TreeParams(10), TreeParams(10)]
    pvs_european = [13.282310163583209, 10.1791573726, 12.967550694010356, 8.157177, 13.989518, 9.125927]
    europeanContracts = [EuropeanContract(und, PutCallFwd.CALL, LongShort.LONG, strike, expiry),
                         EuropeanContract(und, PutCallFwd.CALL, LongShort.LONG, strike, expiry),
                         EuropeanContract(und, PutCallFwd.CALL, LongShort.LONG, strike, expiry),
                         EuropeanContract(und, PutCallFwd.PUT, LongShort.LONG, strike, expiry),
                         EuropeanContract(und, PutCallFwd.CALL, LongShort.LONG, strike, expiry),
                         EuropeanContract(und, PutCallFwd.PUT, LongShort.LONG, strike, expiry)]

    @pytest.mark.parametrize('param, contract, expected_pv', zip(europeanParams, europeanContracts, pvs_european))
    def test_european_tree_pricer(self, param, contract, expected_pv):
        pricer = EuropeanTreePricer(contract, self.model, param)
        pv = pricer.calc_fair_value()
        assert pv == pytest.approx(expected_pv, rel=1e-2)

    americanParams = [TreeParams(10, np.nan, 1.2, 0.8), TreeParams(10), TreeParams(10), TreeParams(10), TreeParams(100)]
    pvs_american = [27.45, 13.989518, -13.989518, 9.730767, 14.218382]
    americanContracts = [AmericanContract(und, PutCallFwd.CALL, LongShort.LONG, strike, expiry),
                         AmericanContract(und, PutCallFwd.CALL, LongShort.LONG, strike, expiry),
                         AmericanContract(und, PutCallFwd.CALL, LongShort.SHORT, strike, expiry),
                         AmericanContract(und, PutCallFwd.PUT, LongShort.LONG, strike, expiry),
                         AmericanContract(und, PutCallFwd.CALL, LongShort.LONG, strike, expiry)]
    @pytest.mark.parametrize('param, contract, expected_pv', zip(americanParams, americanContracts, pvs_american))
    def test_american_tree_pricer(self, param, contract, expected_pv):
        pricer = AmericanTreePricer(contract, self.model, param)
        pv = pricer.calc_fair_value()
        assert pv == pytest.approx(expected_pv, rel=1e-2)


@pytest.mark.parametrize('underlying', [Stock.TEST_COMPANY, Stock.BLUECHIP_BANK])
@pytest.mark.parametrize('ref_strike', np.arange(0.5, 2.0, 0.5))
@pytest.mark.parametrize('expiry', np.arange(0.5, 2.5, 0.5))
@pytest.mark.parametrize('model', [BSVolModel, FlatVolModel])
class TestPutCallParity:
    @pytest.mark.parametrize('greek_method', [GreekMethod.ANALYTIC, GreekMethod.BUMP])
    def test_analytic_pricer(self, underlying, ref_strike, expiry, model, greek_method):
        pricer = dict()
        result = dict()
        mod = model(underlying)
        strike = ref_strike * MarketData.get_spot()[underlying]
        contract = ForwardContract(underlying, LongShort.LONG, strike, expiry)
        pricer['fwd'] = ForwardAnalyticPricer(contract, mod, Params())
        contract = EuropeanContract(underlying, PutCallFwd.CALL, LongShort.LONG, strike, expiry)
        pricer['call'] = EuropeanAnalyticPricer(contract, mod, Params())
        contract = EuropeanContract(underlying, PutCallFwd.PUT, LongShort.SHORT, strike, expiry)
        pricer['put'] = EuropeanAnalyticPricer(contract, mod, Params())
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


class TestEuropeanPDEPricer:
    und = Stock.TEST_COMPANY
    expiry = 2.0
    strike = 0.95 * MarketData.get_spot()[und]
    model = FlatVolModel(und)
    params = [PDEParams(method=BSPDEMethod.EXPLICIT), PDEParams(method=BSPDEMethod.IMPLICIT), PDEParams(method=BSPDEMethod.CRANK_NICOLSON)]
    contract = EuropeanContract(und, PutCallFwd.CALL, LongShort.LONG, strike, expiry)
    pvs = [19.559315913934707, 19.557755216197634, 19.558594355475716]

    @pytest.mark.parametrize('param, expected_pv', zip(params, pvs))
    def test_tree_pricer(self, param, expected_pv):
        pricer = EuropeanPDEPricer(self.contract, self.model, param)
        pv = pricer.calc_fair_value()
        assert pv == pytest.approx(expected_pv)


class TestAsianPricer:
    und = Stock.TEST_COMPANY
    model = BSVolModel(und)
    spot = model.spot
    exp = 1
    num_mon = 100
    moneyness_list = [i / 10 for i in range(7, 14)]
    pvs = [31.05527716868351, 22.010208533350703, 14.092967347218911, 8.055655385255257, 4.114116236896775,
           1.8952610528938865, 0.7982257390694725]
    method = GreekMethod.BUMP
    gammas = [0.00173634060894301, 0.007193813056463938, 0.01565368467101358, 0.02149612527253897,
             0.021052529737006775, 0.016001929107746227, 0.010026197274399684]

    @pytest.mark.parametrize('moneyness, expected_pv', zip(moneyness_list, pvs))
    def test_asian_pricer_pv(self, moneyness, expected_pv):
        contract = AsianContract(self.und, PutCallFwd.CALL, LongShort.LONG, self.spot * moneyness, self.exp, self.num_mon)
        pricer = AsianMomentMatchingPricer(contract, self.model, Params())
        pv = pricer.calc_fair_value()
        assert pv == pytest.approx(expected_pv)

    @pytest.mark.parametrize('moneyness, expected_gamma', zip(moneyness_list, gammas))
    def test_asian_pricer_gamma(self, moneyness, expected_gamma):
        contract = AsianContract(self.und, PutCallFwd.CALL, LongShort.LONG, self.spot * moneyness, self.exp, self.num_mon)
        pricer = AsianMomentMatchingPricer(contract, self.model, Params())
        pv = pricer.calc_gamma(self.method)
        assert pv == pytest.approx(expected_gamma)

