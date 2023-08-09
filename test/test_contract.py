import pytest
import numpy as np
from src.contract import *
from src.enums import *
from src.market_data import *

MarketData.initialize()


class TestContractProperties:
    underlying = Stock.EXAMPLE1
    derivative_type = PutCallFwd.CALL
    long_short = LongShort.LONG
    ref_spot = MarketData.get_initial_spot()[underlying]
    strike = 1.0 * ref_spot
    expiry = 1.0

    @pytest.mark.parametrize('contract_property',
                             ['underlying', 'type', 'long_short', 'strike', 'expiry', 'observations'])
    def test_contract_to_dict(self, contract_property):
        contract_param_map = {
            'ForwardContract': [self.underlying, self.long_short, self.strike, self.expiry],
            'EuropeanContract': [self.underlying, self.derivative_type, self.long_short, self.strike, self.expiry],
            'AmericanContract': [self.underlying, self.derivative_type, self.long_short, self.strike, self.expiry],
            'EuropeanDigitalContract': [self.underlying, self.derivative_type, self.long_short, self.strike,
                                        self.expiry],
            'EuropeanBarrierContract': [self.underlying, self.derivative_type, self.long_short, self.strike,
                                        12, self.expiry, 1.05 * self.ref_spot, UpDown.UP, InOut.OUT],
            'AsianContract': [self.underlying, self.derivative_type, self.long_short, self.strike, self.expiry, 12],
        }
        for class_name, params in contract_param_map.items():
            contract = globals()[class_name](*params)
            assert contract_property in contract.to_dict().keys()


@pytest.mark.parametrize('spot', np.arange(0.5, 2, 0.5))
@pytest.mark.parametrize('strike', np.arange(0.5, 2, 0.5))
class TestPayoff:
    underlying = Stock.EXAMPLE1
    ref_spot = MarketData.get_initial_spot()[underlying]
    long_short = LongShort.LONG
    expiry = 1.0

    def test_forward_payoff(self, spot, strike):
        spot = spot * self.ref_spot
        strike = strike * self.ref_spot
        contract = ForwardContract(self.underlying, self.long_short, strike, self.expiry)
        obs = {round(self.expiry, contract.TIMELINE_DIGITS): spot}
        assert contract.payoff(obs) == pytest.approx(spot - strike)

    def test_call_payoff(self, spot, strike):
        spot = spot * self.ref_spot
        strike = strike * self.ref_spot
        contract = EuropeanContract(self.underlying, PutCallFwd.CALL, self.long_short, strike, self.expiry)
        obs = {round(self.expiry, contract.TIMELINE_DIGITS): spot}
        assert contract.payoff(obs) == pytest.approx(max(spot - strike, 0))

    def test_put_payoff(self, spot, strike):
        spot = spot * self.ref_spot
        strike = strike * self.ref_spot
        contract = EuropeanContract(self.underlying, PutCallFwd.PUT, self.long_short, strike, self.expiry)
        obs = {round(self.expiry, contract.TIMELINE_DIGITS): spot}
        assert contract.payoff(obs) == pytest.approx(max(strike - spot, 0))


@pytest.mark.parametrize('strike', np.arange(0.5, 2, 0.5))
@pytest.mark.parametrize('long_short', [LongShort.LONG, LongShort.SHORT])
class TestGenericPayoff:
    underlying = Stock.EXAMPLE1
    ref_spot = MarketData.get_initial_spot()[underlying]
    expiry = 2.0

    @pytest.mark.parametrize('spot', np.arange(0.5, 2, 0.5))
    def test_forward_payoff(self, long_short, strike, spot):
        spot = spot * self.ref_spot
        strike = strike * self.ref_spot
        contract = ForwardContract(self.underlying, long_short, strike, self.expiry)
        generic = contract.convert_to_generic()
        obs = {round(self.expiry, contract.TIMELINE_DIGITS): spot}
        assert contract.payoff(obs) == pytest.approx(generic.payoff(obs))

    @pytest.mark.parametrize('spot', np.arange(0.5, 2, 0.5))
    @pytest.mark.parametrize('put_call', [PutCallFwd.CALL, PutCallFwd.PUT])
    def test_european_payoff(self, put_call, long_short, strike, spot):
        spot = spot * self.ref_spot
        strike = strike * self.ref_spot
        contract = EuropeanContract(self.underlying, put_call, long_short, strike, self.expiry)
        generic = contract.convert_to_generic()
        obs = {round(self.expiry, contract.TIMELINE_DIGITS): spot}
        assert contract.payoff(obs) == pytest.approx(generic.payoff(obs))

    @pytest.mark.parametrize('spot', np.arange(0.5, 2, 0.5))
    @pytest.mark.parametrize('put_call', [PutCallFwd.CALL, PutCallFwd.PUT])
    def test_american_payoff(self, put_call, long_short, strike, spot):
        spot = spot * self.ref_spot
        strike = strike * self.ref_spot
        contract = AmericanContract(self.underlying, put_call, long_short, strike, self.expiry)
        generic = contract.convert_to_generic()
        obs = {round(self.expiry, contract.TIMELINE_DIGITS): spot}
        assert contract.payoff(obs) == pytest.approx(generic.payoff(obs))

    @pytest.mark.parametrize('spot', np.arange(0.5, 2, 0.5))
    @pytest.mark.parametrize('put_call', [PutCallFwd.CALL, PutCallFwd.PUT])
    def test_european_digital_payoff(self, put_call, long_short, strike, spot):
        spot = spot * self.ref_spot
        strike = strike * self.ref_spot
        contract = EuropeanDigitalContract(self.underlying, put_call, long_short, strike, self.expiry)
        generic = contract.convert_to_generic()
        obs = {round(self.expiry, contract.TIMELINE_DIGITS): spot}
        assert contract.payoff(obs) == pytest.approx(generic.payoff(obs))

    @pytest.mark.parametrize('spot', [np.arange(1.2, 0.7, -0.05), np.arange(1, 2, 0.1)])
    @pytest.mark.parametrize('put_call', [PutCallFwd.CALL, PutCallFwd.PUT])
    def test_asian_payoff(self, put_call, long_short, strike, spot):
        spot = spot * self.ref_spot
        strike = strike * self.ref_spot
        num_mon = len(spot)
        contract = AsianContract(self.underlying, put_call, long_short, strike, self.expiry, num_mon)
        generic = contract.convert_to_generic()
        obs = {round(((i + 1) / num_mon) * self.expiry, contract.TIMELINE_DIGITS): spot[i] for i in range(num_mon)}
        assert contract.payoff(obs) == pytest.approx(generic.payoff(obs))

    @pytest.mark.parametrize('spot', [np.arange(1.2, 0.7, -0.05), np.arange(1, 2, 0.1)])
    @pytest.mark.parametrize('put_call', [PutCallFwd.CALL, PutCallFwd.PUT])
    @pytest.mark.parametrize('barrier', np.arange(0.9, 1.2, 0.2))
    @pytest.mark.parametrize('up_down', [UpDown.DOWN, UpDown.UP])
    @pytest.mark.parametrize('in_out', [InOut.IN, InOut.OUT])
    def test_european_barrier_payoff(self, put_call, long_short, strike, barrier, up_down, in_out, spot):
        spot = spot * self.ref_spot
        strike = strike * self.ref_spot
        barrier = barrier * self.ref_spot
        num_mon = len(spot)
        contract = EuropeanBarrierContract(
            self.underlying, put_call, long_short, strike, self.expiry, num_mon, barrier, up_down, in_out)
        generic = contract.convert_to_generic()
        obs = {round(((i + 1) / num_mon) * self.expiry, contract.TIMELINE_DIGITS): spot[i] for i in range(num_mon)}
        assert contract.payoff(obs) == pytest.approx(generic.payoff(obs))
