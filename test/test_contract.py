import pytest
import numpy as np
from src.contract import *
from src.enums import *
from src.market_data import *

MarketData.initialize()


class TestContractProperties:
    underlying = Stock.TEST_COMPANY
    derivative_type = PutCallFwd.CALL
    long_short = LongShort.LONG
    ref_spot = MarketData.get_spot()[underlying]
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
    underlying = Stock.TEST_COMPANY
    ref_spot = MarketData.get_spot()[underlying]
    long_short = LongShort.LONG
    expiry = 1.0

    def test_forward_payoff(self, spot, strike):
        spot = spot * self.ref_spot
        strike = strike * self.ref_spot
        contract = ForwardContract(self.underlying, self.long_short, strike, self.expiry)
        obs = {round(self.expiry, contract.timeline_digits): spot}
        assert contract.payoff(obs) == pytest.approx(spot - strike)

    def test_call_payoff(self, spot, strike):
        spot = spot * self.ref_spot
        strike = strike * self.ref_spot
        contract = EuropeanContract(self.underlying, PutCallFwd.CALL, self.long_short, strike, self.expiry)
        obs = {round(self.expiry, contract.timeline_digits): spot}
        assert contract.payoff(obs) == pytest.approx(max(spot - strike, 0))

    def test_put_payoff(self, spot, strike):
        spot = spot * self.ref_spot
        strike = strike * self.ref_spot
        contract = EuropeanContract(self.underlying, PutCallFwd.PUT, self.long_short, strike, self.expiry)
        obs = {round(self.expiry, contract.timeline_digits): spot}
        assert contract.payoff(obs) == pytest.approx(max(strike - spot, 0))
