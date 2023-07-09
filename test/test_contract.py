import numpy as np
import pytest
from src.contract import *
from src.enums import *


class TestContractProperties:
    underlying = Stock.EXAMPLE1
    derivative_type = PutCallFwd.CALL
    longshort = LongShort.LONG
    strike = 1.0
    expiry = 1.0

    @pytest.mark.parametrize('contract_property', ['underlying', 'type', 'longshort', 'strike', 'expiry'])
    def test_contract_to_dict(self, contract_property):
        contract_param_map = {
            'ForwardContract': [self.underlying, self.longshort, self.strike, self.expiry],
            'EuropeanContract': [self.underlying, self.derivative_type, self.longshort, self.strike, self.expiry],
            'AmericanContract': [self.underlying, self.derivative_type, self.longshort, self.strike, self.expiry],
            'EuropeanDigitalContract': [self.underlying, self.derivative_type, self.longshort, self.strike, self.expiry],
        }
        for class_name, params in contract_param_map.items():
            contract = globals()[class_name](*params)
            assert contract_property in contract.to_dict().keys()


@pytest.mark.parametrize('spot', np.arange(0, 2, 0.25))
@pytest.mark.parametrize('strike', np.arange(0, 2, 0.25))
class TestPayoff:
    underlying = Stock.EXAMPLE1
    longshort = LongShort.LONG
    expiry = 1.0

    def test_forward_payoff(self, spot, strike):
        contract = ForwardContract(self.underlying, self.longshort, strike, self.expiry)
        assert contract.payoff(spot) == pytest.approx(spot - strike)

    def test_call_payoff(self, spot, strike):
        contract = EuropeanContract(self.underlying, PutCallFwd.CALL, self.longshort, strike, self.expiry)
        assert contract.payoff(spot) == pytest.approx(max(spot - strike, 0))

    def test_put_payoff(self, spot, strike):
        contract = EuropeanContract(self.underlying, PutCallFwd.PUT, self.longshort, strike, self.expiry)
        assert contract.payoff(spot) == pytest.approx(max(strike - spot, 0))


@pytest.mark.parametrize('spot', np.arange(-1, 3, 0.5))
@pytest.mark.parametrize('strike', np.arange(-1, 3, 0.5))
@pytest.mark.parametrize('longshort', [LongShort.LONG, LongShort.SHORT])
class TestGenericPayoff_Fwd:
    expiry = 2

    def test_forward_payoff(self, longshort, spot, strike):
        trade = ForwardContract('Apple', longshort, strike, self.expiry)
        generic_trade = trade.convert_to_generic()
        assert trade.payoff(spot) == pytest.approx(generic_trade.payoff(spot))


@pytest.mark.parametrize('spot', np.arange(-1, 3, 0.5))
@pytest.mark.parametrize('strike', np.arange(-1, 3, 0.5))
@pytest.mark.parametrize('longshort', [LongShort.LONG, LongShort.SHORT])
@pytest.mark.parametrize('putcall', [PutCallFwd.CALL, PutCallFwd.PUT])
class TestGenericPayoff_Eu_Am_EuDig:
    expiry = 2.0

    def test_European_payoff(self, putcall, longshort, spot, strike):
        trade = EuropeanContract('OTP', putcall, longshort, strike, self.expiry)
        generic_trade = trade.convert_to_generic()
        assert trade.payoff(spot) == pytest.approx(generic_trade.payoff(spot))

    def test_American_payoff(self, putcall, longshort, spot, strike):
        trade = AmericanContract('Tesla', putcall, longshort, strike, self.expiry)
        generic_trade = trade.convert_to_generic()
        assert trade.payoff(spot) == pytest.approx(generic_trade.payoff(spot))

    def test_EuropeanDigital_payoff(self, putcall, longshort, spot, strike):
        trade = EuropeanDigitalContract('Mol', putcall, longshort, strike, self.expiry)
        generic_trade = trade.convert_to_generic()
        assert trade.payoff(spot) == pytest.approx(generic_trade.payoff(spot))



@pytest.mark.parametrize('strike', np.arange(-1, 3, 0.5))
@pytest.mark.parametrize('longshort', [LongShort.LONG, LongShort.SHORT])
@pytest.mark.parametrize('putcall', [PutCallFwd.CALL, PutCallFwd.PUT])
class TestGenericPayoff_Asian_Barr:
    spot = np.arange(-1, 3, 0.5) 
    expiry = 2.0

    def test_Asian_payoff(self, putcall, longshort, strike):
        trade = AsianContract('Microsoft', putcall, longshort, strike, self.expiry)
        generic_trade = trade.convert_to_generic()
        assert trade.payoff(self.spot) == pytest.approx(generic_trade.payoff(self.spot))

    @pytest.mark.parametrize('updown', [UpDown.DOWN, UpDown.UP])
    @pytest.mark.parametrize('inout', [InOut.IN, InOut.OUT])
    @pytest.mark.parametrize('barrier', np.arange(-1, 3, 0.5))
    def test_EuropeanBarrier_payoff(self, putcall, longshort, strike, updown, inout, barrier):
        trade = EuropeanBarrierContract('Deutsche Bank', putcall, longshort, strike, self.expiry,
                                        barrier, updown, inout)
        generic_trade = trade.convert_to_generic()
        assert trade.payoff(self.spot) == pytest.approx(generic_trade.payoff(self.spot))

