import numpy as np
import pytest
from pricer import *
from scipy.stats import norm


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

        # Homework 3 - Test European Digital Option Fair Value

@pytest.mark.parametrize(
            "underlying",
            [Stock.BLUECHIP_BANK, Stock.TIPTOP_SOLUTIONS]
        )
@pytest.mark.parametrize("strike_over_spot", [0.8, 1.0, 1.2])
@pytest.mark.parametrize("expiry", [0.5, 1.0, 2.0])
@pytest.mark.parametrize(
            "derivative_type",
            [PutCallFwd.CALL, PutCallFwd.PUT]
        )
def test_digital_option_fair_value(
                underlying, strike_over_spot, expiry, derivative_type
        ):
            model = MarketModel(underlying)
            params = Params()

            spot = model.spot
            strike = strike_over_spot * spot
            vol = model.vol
            rate = model.risk_free_rate

            contract = EuropeanDigitalContract(
                underlying,
                derivative_type,
                LongShort.LONG,
                strike,
                expiry
            )

            pricer = EuropeanDigitalAnalyticPricer(
                contract, model, params
            )

            actual_price = pricer.calc_fair_value()

            d2 = (
                         np.log(spot / strike)
                         + (rate - 0.5 * vol ** 2) * expiry
                 ) / (vol * np.sqrt(expiry))

            discount_factor = np.exp(-rate * expiry)

            if derivative_type == PutCallFwd.CALL:
                expected_price = discount_factor * norm.cdf(d2)
            else:
                expected_price = discount_factor * norm.cdf(-d2)

            assert actual_price == pytest.approx(
                expected_price, abs=1e-10
            )

            # Homework 3 - Numerical validation of digital Greeks

@pytest.mark.parametrize("option_type", [
                PutCallFwd.CALL, PutCallFwd.PUT
            ])
def test_digital_greeks(option_type):
                underlying = Stock.BLUECHIP_BANK
                model = MarketModel(underlying)
                params = Params()

                S = model.spot
                K = S
                vol = model.vol
                r = model.risk_free_rate
                T = 1.0

                contract = EuropeanDigitalContract(
                    underlying, option_type, LongShort.LONG, K, T
                )
                pricer = EuropeanDigitalAnalyticPricer(
                    contract, model, params
                )

                eta = 1 if option_type == PutCallFwd.CALL else -1

                def price(s=S, sigma=vol, rate=r, tau=T):
                    d2 = (
                                 np.log(s / K) + (rate - 0.5 * sigma ** 2) * tau
                         ) / (sigma * np.sqrt(tau))
                    return np.exp(-rate * tau) * norm.cdf(eta * d2)

                hs = S * 0.001
                h = 0.0001

                numerical = {
                    "delta": (price(s=S + hs) - price(s=S - hs)) / (2 * hs),
                    "gamma": (
                                     price(s=S + hs) - 2 * price() + price(s=S - hs)
                             ) / hs ** 2,
                    "vega": (
                                    price(sigma=vol + h) - price(sigma=vol - h)
                            ) / (2 * h),
                    "theta": (
                                     price(tau=T - h) - price(tau=T + h)
                             ) / (2 * h),
                    "rho": (
                                   price(rate=r + h) - price(rate=r - h)
                           ) / (2 * h),
                }

                analytic = {
                    "delta": pricer.calc_delta(),
                    "gamma": pricer.calc_gamma(),
                    "vega": pricer.calc_vega(),
                    "theta": pricer.calc_theta(),
                    "rho": pricer.calc_rho(),
                }

                for greek in analytic:
                    assert analytic[greek] == pytest.approx(
                        numerical[greek], rel=0.002, abs=0.00002
                    )


