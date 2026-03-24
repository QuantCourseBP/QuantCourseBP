import numpy as np
import matplotlib.pyplot as plt

from enums import Stock, PutCallFwd, LongShort
from model import MarketModel
from contract import EuropeanContract, EuropeanDigitalContract
from pricer import Params, EuropeanAnalyticPricer, EuropeanDigitalAnalyticPricer


def digital_call_price(spot: float, strike: float, expiry: float, underlying: Stock) -> float:
    params = Params()
    model = MarketModel(underlying)
    model.spot = spot

    contract = EuropeanDigitalContract(
        underlying=underlying,
        derivative_type=PutCallFwd.CALL,
        long_short=LongShort.LONG,
        strike=strike,
        expiry=expiry
    )

    pricer = EuropeanDigitalAnalyticPricer(contract, model, params)
    return pricer.calc_fair_value()
def scaled_bull_spread_price(spot: float, strike: float, expiry: float, eps: float, underlying: Stock) -> float:
    params = Params()
    model = MarketModel(underlying)
    model.spot = spot

    # 1. Normál európai Call a K (strike) áron
    call_long = EuropeanContract(
        underlying=underlying,
        derivative_type=PutCallFwd.CALL,
        long_short=LongShort.LONG,  # Mindkettőt LONG-ként árazzuk!
        strike=strike,              # Induljunk pontosan a strike-ról
        expiry=expiry
    )

    # 2. Normál európai Call a K + eps áron
    call_short = EuropeanContract(
        underlying=underlying,
        derivative_type=PutCallFwd.CALL,
        long_short=LongShort.LONG,  # Ezt is LONG-ként árazzuk!
        strike=strike + eps,        # Eltolás epszilonggal (Előrefelé vett differencia)
        expiry=expiry
    )

    pricer_long = EuropeanAnalyticPricer(call_long, model, params)
    pricer_short = EuropeanAnalyticPricer(call_short, model, params)

    # Manuális kivonás: Long Call(K) - Long Call(K+eps), majd osztás eps-el
    return (pricer_long.calc_fair_value() - pricer_short.calc_fair_value()) / eps


def main():
    underlying = Stock.BLUECHIP_BANK
    expiry = 1.0

    base_model = MarketModel(underlying)
    strike = base_model.spot

    # Finomabb felbontás a strike körül
    spots = np.linspace(0.5 * strike, 1.5 * strike, 500)

    digital_prices = [
        digital_call_price(spot=s, strike=strike, expiry=expiry, underlying=underlying)
        for s in spots
    ]

    # Kicsit nagyobb eps értékek, hogy a "durva" becslés vizuálisan elváljon
    eps_list = [0.15 * strike, 0.08 * strike, 0.03 * strike, 0.005 * strike]

    plt.figure(figsize=(10, 6))
    
    # A digitális opciót vastagabb, domináns vonallal rajzoljuk
    plt.plot(spots, digital_prices, label='Digital call price (Theoretical)', linewidth=3, color='black')

    for eps in eps_list:
        bull_prices = [
            scaled_bull_spread_price(spot=s, strike=strike, expiry=expiry, eps=eps, underlying=underlying)
            for s in spots
        ]
        plt.plot(spots, bull_prices, label=f'Scaled bull spread, eps={eps:.2f}', linestyle='--')

    plt.axvline(strike, color='red', linestyle=':', linewidth=1.5, label=f'Strike = {strike:.2f}')
    
    # KULCSFONTOSSÁGÚ: Ránagyítunk a Strike körüli részre, mert ott történik a varázslat
    plt.xlim(0.85 * strike, 1.15 * strike)
    plt.ylim(-0.1, 1.1) # A digitális opció ára jellemzően 0 és e^-rT között mozog

    plt.xlabel('Spot Price (S)')
    plt.ylabel('Fair value')
    plt.title('Digital Call as Limit of Scaled Bull Spread (Zoomed around Strike)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()



