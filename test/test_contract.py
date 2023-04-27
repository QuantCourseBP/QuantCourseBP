import numpy as np
import pytest
from src.contract import *
from src.enums import *


class TestDiagnostics:
    underlying = Stock.EXAMPLE1
    derivative_type = PutCallFwd.CALL
    strike = 1.0
    expiry = 1.0

    @pytest.mark.parametrize('contract_property', ['underlying', 'type', 'strike', 'expiry'])
    def test_contract_to_dict(self, contract_property):
        contract_param_map = {
            'ForwardContract': [self.underlying, self.strike, self.expiry],
            'EuropeanContract': [self.underlying, self.derivative_type, self.strike, self.expiry],
            'AmericanContract': [self.underlying, self.derivative_type, self.strike, self.expiry],
            'EuropeanDigitalContract': [self.underlying, self.derivative_type, self.strike, self.expiry],
        }
        for class_name, params in contract_param_map.items():
            contract = globals()[class_name](*params)
            assert contract_property in contract.to_dict().keys()


@pytest.mark.parametrize('spot', np.arange(0, 2, 0.25))
@pytest.mark.parametrize('strike', np.arange(0, 2, 0.25))
class TestPayoff:
    underlying = Stock.EXAMPLE1
    expiry = 1.0

    def test_forward_payoff(self, spot, strike):
        contract = ForwardContract(self.underlying, strike, self.expiry)
        assert contract.payoff(spot) == spot - strike

    def test_call_payoff(self, spot, strike):
        contract = EuropeanContract(self.underlying, PutCallFwd.CALL, strike, self.expiry)
        assert contract.payoff(spot) == max(spot - strike, 0)

    def test_put_payoff(self, spot, strike):
        contract = EuropeanContract(self.underlying, PutCallFwd.PUT, strike, self.expiry)
        assert contract.payoff(spot) == max(strike - spot, 0)