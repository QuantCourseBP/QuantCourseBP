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
    def test_european_pde_pricer(self, param, expected_pv):
        pricer = EuropeanPDEPricer(self.contract, self.model, param)
        pv = pricer.calc_fair_value()
        assert pv == pytest.approx(expected_pv)


class TestAmericanPDEPricer:
    und = Stock.TEST_COMPANY
    expiry = 2.0
    strike = 0.95 * MarketData.get_spot()[und]
    model = FlatVolModel(und)
    params = [PDEParams(method=BSPDEMethod.EXPLICIT), PDEParams(method=BSPDEMethod.IMPLICIT), PDEParams(method=BSPDEMethod.CRANK_NICOLSON)]
    contract = AmericanContract(und, PutCallFwd.PUT, LongShort.LONG, strike, expiry)
    pvs = [6.281464959775393, 6.279306011571068, 6.2803811513002366]

    @pytest.mark.parametrize('param, expected_pv', zip(params, pvs))
    def test_american_pde_pricer(self, param, expected_pv):
        pricer = AmericanPDEPricer(self.contract, self.model, param)
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

    @pytest.mark.parametrize('moneyness, expected_pv, expected_gamma', zip(moneyness_list, pvs, gammas))
    def test_asian_pricer(self, moneyness, expected_pv, expected_gamma):
        contract = AsianContract(self.und, PutCallFwd.CALL, LongShort.LONG, self.spot * moneyness, self.exp, self.num_mon)
        pricer = AsianMomentMatchingPricer(contract, self.model, Params())
        pv = pricer.calc_fair_value()
        assert pv == pytest.approx(expected_pv)
        gamma = pricer.calc_gamma(self.method)
        assert gamma == pytest.approx(expected_gamma)


class TestBarrierPricer:
    und = Stock.TEST_COMPANY
    model = BSVolModel(und)
    spot = model.spot
    exp = 1
    num_mon = 100
    barrier = 90
    up_down = UpDown.DOWN
    in_out = InOut.IN
    moneyness_list = [i / 10 for i in range(7, 14)]
    pvs = [17.590846234380994, 11.87723169824169, 7.704228407261273, 4.838479479055546, 2.963276479066368,
           1.7806959944654814, 1.0552699267786232]
    method = GreekMethod.BUMP
    gammas = [0.021967522460151656, 0.02151691105424014, 0.019387672150170232, 0.016169753179283042,
              0.012638449622107917, 0.00937559239729513, 0.006674222683586084]
    params_MC = MCParams(seed=1, num_of_path=10, tenor_frequency=1)
    pvs_mc = [(21.085478746034322, 10.046055843189333, 32.124901648879316),
              (14.426872774529325, 5.560535431385187, 23.29321011767346),
              (8.30405270019804, 1.4721102345162524, 15.135995165879828),
              (4.238539511857375, -0.047198181906164836, 8.524277205620917),
              (1.182238374471723, -1.016145787933318, 3.380622536876764),
              (0.17522259572646742, -0.16821369189740873, 0.5186588833503436),
              (0.0, 0.0, 0.0)]

    @pytest.mark.parametrize('moneyness, expected_pv, expected_gamma', zip(moneyness_list, pvs, gammas))
    def test_barrier_analytic_pricer(self, moneyness, expected_pv, expected_gamma):
        contract = EuropeanBarrierContract(self.und, PutCallFwd.CALL, LongShort.LONG, self.spot * moneyness, self.exp,
                                           self.num_mon, self.barrier, self.up_down, self.in_out)
        pricer = BarrierAnalyticPricer(contract, self.model, Params())
        pv = pricer.calc_fair_value()
        assert pv == pytest.approx(expected_pv)
        gamma = pricer.calc_gamma(self.method)
        assert gamma == pytest.approx(expected_gamma)

    @pytest.mark.parametrize('moneyness, expected_pv', zip(moneyness_list, pvs_mc))
    def test_barrier_MC_pricer(self, moneyness, expected_pv):
        contract = EuropeanBarrierContract(self.und, PutCallFwd.CALL, LongShort.LONG, self.spot * moneyness, self.exp,
                                           self.num_mon, self.barrier, self.up_down, self.in_out)
        pricer = GenericMCPricer(contract, self.model, self.params_MC)
        pv = pricer.calc_fair_value_with_ci()
        assert pv[0] == pytest.approx(expected_pv[0])
        for i in range(1):
            assert pv[1][i] == pytest.approx(expected_pv[i+1])





