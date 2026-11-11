import pytest
from src.pricer import *

MarketData.initialize()


@pytest.mark.parametrize('fd', list(FiniteDifferenceMethod))
@pytest.mark.parametrize('bump_type', list(BumpType))
@pytest.mark.parametrize('kind', [PutCallFwd.CALL, PutCallFwd.PUT])
def test_bump_greeks_close_to_analytic(fd, bump_type, kind):
    model = BSVolModel(Stock.BLUECHIP_BANK)
    contract = EuropeanContract(Stock.BLUECHIP_BANK, kind, LongShort.LONG, model.spot, 1.0)
    pricer = EuropeanAnalyticPricer(contract, model, Params())
    Pricer.fd_method, Pricer.bump_type = fd, bump_type
    Pricer.spot_bump_size = 1e-4 if bump_type == BumpType.RELATIVE else 1e-4 * model.spot
    try:
        assert pricer.calc_delta(GreekMethod.BUMP) == pytest.approx(pricer.calc_delta(GreekMethod.ANALYTIC), abs=1e-3)
        assert pricer.calc_gamma(GreekMethod.BUMP) == pytest.approx(pricer.calc_gamma(GreekMethod.ANALYTIC), rel=1e-2)
    finally:
        Pricer.fd_method, Pricer.bump_type, Pricer.spot_bump_size = (
            FiniteDifferenceMethod.CENTRAL, BumpType.RELATIVE, 0.01)
