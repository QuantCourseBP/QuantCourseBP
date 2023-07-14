import numpy as np
import pytest
from src.pricer import *

class TestTreePricer:
    MarketData.initialize()
    
    model = FlatVolModel(Stock.EXAMPLE1)
    expiry = 1.0
    moneyness = 1.0
    contract = EuropeanContract(Stock.EXAMPLE1, PutCallFwd.CALL, moneyness, expiry)
    
    params = [TreeParams(expiry, moneyness, 2, 1.2, 0.8), TreeParams(expiry, moneyness, 2)]
    pvs = [0.0798231016, 0.104734926]

    @pytest.mark.parametrize('param, expected_pv', zip(params, pvs))
    def test_tree_pricer(self, param, expected_pv):
        pricer = GenericTreePricer(TestTreePricer.contract, TestTreePricer.model, param)
        pv = pricer.calc_fair_value()
        spot_tree = pricer._tree_method._spot_tree
        assert pv == pytest.approx(expected_pv)

