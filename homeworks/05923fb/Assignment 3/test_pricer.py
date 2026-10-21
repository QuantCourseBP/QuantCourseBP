import numpy as np
import pytest
from pricer import *


# TASK:
# 1. Create put-call parity test for fair value and all greeks using various underlying, strike, expiry.
# ASSIGNMENT 3:
# a. Implement unit test for fair value of EuropeanDigitalAnalyticPricer -> TestEuropeanDigitalAnalyticPricer.


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


def create_digital_pricer(underlying: Stock, derivative_type: PutCallFwd, long_short: LongShort,
                          strike_over_spot: float, expiry: float) -> EuropeanDigitalAnalyticPricer:
    model = MarketModel(underlying)
    contract = EuropeanDigitalContract(underlying, derivative_type, long_short, strike_over_spot * model.spot, expiry)
    return EuropeanDigitalAnalyticPricer(contract, model, Params())


def create_european_pricer(underlying: Stock, derivative_type: PutCallFwd, strike: float,
                           expiry: float) -> EuropeanAnalyticPricer:
    contract = EuropeanContract(underlying, derivative_type, LongShort.LONG, strike, expiry)
    return EuropeanAnalyticPricer(contract, MarketModel(underlying), Params())


@pytest.mark.parametrize('underlying', [Stock.BLUECHIP_BANK, Stock.TIPTOP_SOLUTIONS])
@pytest.mark.parametrize('strike_over_spot', [0.5, 0.9, 1.0, 1.1, 1.5])
@pytest.mark.parametrize('expiry', [0.25, 1.0, 2.5])
class TestEuropeanDigitalAnalyticPricer:
    def test_fair_value_put_call_parity(self, underlying, strike_over_spot, expiry):
        # digital call + digital put pays 1 almost surely (except on S_T = K), so it is worth the discount factor
        call = create_digital_pricer(underlying, PutCallFwd.CALL, LongShort.LONG, strike_over_spot, expiry)
        put = create_digital_pricer(underlying, PutCallFwd.PUT, LongShort.LONG, strike_over_spot, expiry)
        assert call.calc_fair_value() + put.calc_fair_value() == pytest.approx(call.model.calc_df(expiry), abs=1e-12)

    @pytest.mark.parametrize('derivative_type', [PutCallFwd.CALL, PutCallFwd.PUT])
    def test_fair_value_short_is_negative_of_long(self, underlying, strike_over_spot, expiry, derivative_type):
        long = create_digital_pricer(underlying, derivative_type, LongShort.LONG, strike_over_spot, expiry)
        short = create_digital_pricer(underlying, derivative_type, LongShort.SHORT, strike_over_spot, expiry)
        assert short.calc_fair_value() == pytest.approx(-long.calc_fair_value(), abs=1e-12)

    def test_fair_value_is_limit_of_call_spread(self, underlying, strike_over_spot, expiry):
        # digital call = -dC/dK, approximated by a narrow centered call spread priced with EuropeanAnalyticPricer;
        # truncation error O(h^2), rounding error O(eps / h): both ~1e-10 or below for h = 1e-5 * K
        digital = create_digital_pricer(underlying, PutCallFwd.CALL, LongShort.LONG, strike_over_spot, expiry)
        strike = digital.contract.strike
        h = 1e-5 * strike
        call_low = create_european_pricer(underlying, PutCallFwd.CALL, strike - h, expiry).calc_fair_value()
        call_high = create_european_pricer(underlying, PutCallFwd.CALL, strike + h, expiry).calc_fair_value()
        assert digital.calc_fair_value() == pytest.approx((call_low - call_high) / (2 * h), abs=1e-9)

    @pytest.mark.parametrize('derivative_type', [PutCallFwd.CALL, PutCallFwd.PUT])
    @pytest.mark.parametrize('long_short', [LongShort.LONG, LongShort.SHORT])
    def test_greeks_against_finite_differences(self, underlying, strike_over_spot, expiry, derivative_type,
                                               long_short):
        pricer = create_digital_pricer(underlying, derivative_type, long_short, strike_over_spot, expiry)
        model = pricer.model

        def bumped_fair_value(attribute: str, holder: object, bump: float) -> float:
            original = getattr(holder, attribute)
            setattr(holder, attribute, original + bump)
            value = pricer.calc_fair_value()
            setattr(holder, attribute, original)
            return value

        h_spot, h = 1e-4 * model.spot, 1e-5
        fv = pricer.calc_fair_value()
        fv_spot_up, fv_spot_down = (bumped_fair_value('spot', model, b) for b in (h_spot, -h_spot))
        numeric = {
            'delta': (fv_spot_up - fv_spot_down) / (2 * h_spot),
            'gamma': (fv_spot_up - 2 * fv + fv_spot_down) / h_spot**2,
            'vega': (bumped_fair_value('vol', model, h) - bumped_fair_value('vol', model, -h)) / (2 * h),
            'theta': (bumped_fair_value('valuation_time', pricer, h)
                      - bumped_fair_value('valuation_time', pricer, -h)) / (2 * h),
            'rho': (bumped_fair_value('risk_free_rate', model, h)
                    - bumped_fair_value('risk_free_rate', model, -h)) / (2 * h),
        }
        analytic = {
            'delta': pricer.calc_delta(),
            'gamma': pricer.calc_gamma(),
            'vega': pricer.calc_vega(),
            'theta': pricer.calc_theta(),
            'rho': pricer.calc_rho(),
        }
        assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-9)


class TestEuropeanDigitalContract:
    @pytest.mark.parametrize('derivative_type, spot_at_expiry, expected', [
        (PutCallFwd.CALL, 120.0, 1.0),
        (PutCallFwd.CALL, 80.0, 0.0),
        (PutCallFwd.PUT, 120.0, 0.0),
        (PutCallFwd.PUT, 80.0, 1.0),
    ])
    @pytest.mark.parametrize('long_short, direction', [(LongShort.LONG, 1.0), (LongShort.SHORT, -1.0)])
    def test_payoff(self, derivative_type, spot_at_expiry, expected, long_short, direction):
        contract = EuropeanDigitalContract(Stock.TEST_COMPANY, derivative_type, long_short, 100.0, 1.0)
        assert contract.payoff({1.0: spot_at_expiry}) == direction * expected

    def test_forward_type_is_rejected(self):
        with pytest.raises(ValueError):
            EuropeanDigitalContract(Stock.TEST_COMPANY, PutCallFwd.FWD, LongShort.LONG, 100.0, 1.0)

    def test_pricers_reject_other_contract_types(self):
        model = MarketModel(Stock.BLUECHIP_BANK)
        european = EuropeanContract(Stock.BLUECHIP_BANK, PutCallFwd.CALL, LongShort.LONG, 100.0, 1.0)
        digital = EuropeanDigitalContract(Stock.BLUECHIP_BANK, PutCallFwd.CALL, LongShort.LONG, 100.0, 1.0)
        with pytest.raises(TypeError):
            EuropeanDigitalAnalyticPricer(european, model, Params())
        with pytest.raises(TypeError):
            EuropeanAnalyticPricer(digital, model, Params())
