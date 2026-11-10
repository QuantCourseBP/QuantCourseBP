"""Accuracy, stencil, configuration and state preservation checks."""
import copy
import numpy as np
import pytest
from .pricer import (Pricer, EuropeanAnalyticPricer, ForwardAnalyticPricer, Params)
from .model import MarketData, FlatVolModel, BSVolModel
from .contract import EuropeanContract, ForwardContract
from .enums import Stock, PutCallFwd, LongShort, GreekMethod, FiniteDifferenceMethod, BumpType


@pytest.fixture(scope='module', autouse=True)
def market():
    MarketData.initialize()


def make_pricer(model_type=FlatVolModel, stock=Stock.BLUECHIP_BANK,
                kind=PutCallFwd.CALL, direction=LongShort.LONG, moneyness=1., expiry=1.):
    model = model_type(stock)
    contract = EuropeanContract(stock, kind, direction, model.spot*moneyness, expiry)
    return EuropeanAnalyticPricer(contract, model, Params())


@pytest.mark.parametrize('scheme', list(FiniteDifferenceMethod))
@pytest.mark.parametrize('bump', list(BumpType))
@pytest.mark.parametrize('model_type', [FlatVolModel, BSVolModel])
@pytest.mark.parametrize('stock', [Stock.BLUECHIP_BANK, Stock.TIPTOP_SOLUTIONS])
@pytest.mark.parametrize('kind', [PutCallFwd.CALL, PutCallFwd.PUT])
@pytest.mark.parametrize('direction', list(LongShort))
@pytest.mark.parametrize('moneyness,expiry', [(0.9,.25), (1.,1.), (1.1,2.)])
def test_against_analytic(scheme, bump, model_type, stock, kind, direction, moneyness, expiry):
    p = make_pricer(model_type, stock, kind, direction, moneyness, expiry)
    p.finite_difference_method, p.bump_type = scheme, bump
    p.relative_bump_size = 1e-4
    p.absolute_bump_size = p.model.spot * 1e-4
    p.valuation_time = expiry * .1
    before = (p.model.spot, p.valuation_time, p.model.volgrid.values.copy())
    for method in [p.calc_delta, p.calc_gamma]:
        assert method(GreekMethod.BUMP) == pytest.approx(method(GreekMethod.ANALYTIC), rel=.003, abs=2e-6)
    assert p.model.spot == before[0] and p.valuation_time == before[1]
    np.testing.assert_array_equal(p.model.volgrid.values, before[2])


@pytest.mark.parametrize('scheme', list(FiniteDifferenceMethod))
@pytest.mark.parametrize('order', [1, 2])
def test_exact_stencil_and_offsets(scheme, order):
    class CubicPricer(Pricer):
        seen = []
        def calc_fair_value(self):
            self.seen.append(self.model.spot)
            return self.model.spot**3
    p = CubicPricer(None, FlatVolModel(Stock.TEST_COMPANY), Params())
    p.model.spot = 10.
    p.bump_type = BumpType.ABSOLUTE
    p.absolute_bump_size = .5
    p.finite_difference_method = scheme
    CubicPricer.seen.clear()
    value = (p.calc_delta if order == 1 else p.calc_gamma)(GreekMethod.BUMP)
    expected = {(1, 'FORWARD'):315.25, (1,'BACKWARD'):285.25, (1,'CENTRAL'):300.25,
                (2,'FORWARD'):63., (2,'BACKWARD'):57., (2,'CENTRAL'):60.}
    assert value == pytest.approx(expected[order, scheme.value])
    offsets = {(1,'FORWARD'):[10,10.5], (1,'BACKWARD'):[9.5,10],
               (1,'CENTRAL'):[9.5,10.5], (2,'FORWARD'):[10,10.5,11],
               (2,'BACKWARD'):[9,9.5,10], (2,'CENTRAL'):[9.5,10,10.5]}
    assert CubicPricer.seen == offsets[order, scheme.value]


@pytest.mark.parametrize('scheme', list(FiniteDifferenceMethod))
@pytest.mark.parametrize('order', [1,2])
def test_absolute_relative_equivalence(scheme, order):
    p = make_pricer()
    p.finite_difference_method = scheme
    p.relative_bump_size = .002
    p.absolute_bump_size = .002*p.model.spot
    method = p.calc_delta if order == 1 else p.calc_gamma
    p.bump_type = BumpType.RELATIVE
    relative = method(GreekMethod.BUMP)
    p.bump_type = BumpType.ABSOLUTE
    assert method(GreekMethod.BUMP) == relative


@pytest.mark.parametrize('size', [0., -1., np.nan, np.inf, 1e-30])
def test_invalid_bumps(size):
    p = make_pricer()
    p.relative_bump_size = size
    with pytest.raises(ValueError):
        p.calc_delta(GreekMethod.BUMP)


def test_negative_stencil_and_invalid_settings():
    p = make_pricer()
    p.finite_difference_method = FiniteDifferenceMethod.BACKWARD
    p.relative_bump_size = .6
    with pytest.raises(ValueError):
        p.calc_gamma(GreekMethod.BUMP)
    p.bump_type = 'unknown'
    with pytest.raises(ValueError):
        p.calc_delta(GreekMethod.BUMP)
    p.bump_type = BumpType.RELATIVE
    p.finite_difference_method = 'unknown'
    with pytest.raises(ValueError):
        p.calc_delta(GreekMethod.BUMP)


@pytest.mark.parametrize('scheme', list(FiniteDifferenceMethod))
def test_put_call_parity_and_forward(scheme):
    call = make_pricer()
    put = make_pricer(kind=PutCallFwd.PUT)
    fwd = ForwardAnalyticPricer(ForwardContract(call.contract.underlying, LongShort.LONG,
                              call.contract.strike, call.contract.expiry), call.model, Params())
    for p in [call,put,fwd]:
        p.finite_difference_method = scheme
        p.relative_bump_size = .001
    assert call.calc_delta(GreekMethod.BUMP)-put.calc_delta(GreekMethod.BUMP) == pytest.approx(1.,abs=1e-10)
    assert call.calc_gamma(GreekMethod.BUMP)-put.calc_gamma(GreekMethod.BUMP) == pytest.approx(0.,abs=1e-10)
    assert fwd.calc_delta(GreekMethod.BUMP) == pytest.approx(1.)
    assert fwd.calc_gamma(GreekMethod.BUMP) == pytest.approx(0.,abs=1e-10)


@pytest.mark.parametrize('order', [1,2])
@pytest.mark.parametrize('scheme', list(FiniteDifferenceMethod))
def test_truncation_convergence(order, scheme):
    p = make_pricer(moneyness=1.1)
    p.finite_difference_method = scheme
    method = p.calc_delta if order == 1 else p.calc_gamma
    exact = method(GreekMethod.ANALYTIC)
    errors = []
    for h in [.02,.01]:
        p.relative_bump_size = h
        errors.append(abs(method(GreekMethod.BUMP)-exact))
    expected_ratio = 4 if scheme == FiniteDifferenceMethod.CENTRAL else 2
    assert errors[0]/errors[1] == pytest.approx(expected_ratio,rel=.2)
