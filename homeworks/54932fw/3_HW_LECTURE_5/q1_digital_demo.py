# Homework 3, Q1: the digital option in the library (contract + analytic pricer), running example.
# Run from this folder with the course venv:  python q1_digital_demo.py   (printed output: q1_digital_demo.out)
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
root = next((p for p in [here, *here.parents] if (p / 'src' / 'enums.py').exists()), here)
sys.path[:0] = [str(here), str(root)]               # this folder (library) and the repository (src.enums)

from pricer import *                                # library of Lecture 5 + the NEW classes (Homework 3)

# running example: the market of the library
model = MarketModel(Stock.BLUECHIP_BANK)
strike, expiry = 130.0, 1.0
print(f'market: S = {model.spot}, vol = {model.vol}, r = {model.risk_free_rate};  K = {strike}, T = {expiry}')

# 1. payoffs of the new contract (Definition 1)
long_call = EuropeanDigitalContract(Stock.BLUECHIP_BANK, PutCallFwd.CALL, LongShort.LONG, strike, expiry)
short_put = EuropeanDigitalContract(Stock.BLUECHIP_BANK, PutCallFwd.PUT, LongShort.SHORT, strike, expiry)
for s_T in [129.0, 131.0]:
    print(f'S_T = {s_T}: long digital call pays {long_call.payoff({expiry: s_T})}, '
          f'short digital put pays {short_put.payoff({expiry: s_T})}')

# 2. fair values (Lemma 4) and parity C_d + P_d = DF
pricers = {cp: EuropeanDigitalAnalyticPricer(
               EuropeanDigitalContract(Stock.BLUECHIP_BANK, cp, LongShort.LONG, strike, expiry), model, Params())
           for cp in (PutCallFwd.CALL, PutCallFwd.PUT)}
call, put = (pricers[cp].calc_fair_value() for cp in (PutCallFwd.CALL, PutCallFwd.PUT))
df = model.calc_df(expiry)
d2 = EuropeanAnalyticPricer.calc_d2(model.spot / strike, model.vol, model.risk_free_rate, expiry)
print(f'd2 = {d2:.6f}, DF = {df:.6f}')
print(f'digital call = {call:.6f}, digital put = {put:.6f}, call + put = {call + put:.6f} (DF = {df:.6f})')


# 3. bull spreads with the lecture's vanilla pricer (proof of Lemma 4)
def vanilla_call(k: float) -> float:
    contract = EuropeanContract(Stock.BLUECHIP_BANK, PutCallFwd.CALL, LongShort.LONG, k, expiry)
    return EuropeanAnalyticPricer(contract, model, Params()).calc_fair_value()


for h in [10.0, 1.0, 0.1, 0.01]:
    lower = (vanilla_call(strike) - vanilla_call(strike + h)) / h
    upper = (vanilla_call(strike - h) - vanilla_call(strike)) / h
    print(f'h = {h:5.2f}: {lower:.6f} <= C_d = {call:.6f} <= {upper:.6f}   {lower <= call <= upper}')

# 4. Greeks (Lemma 5) against bump and revaluation
print(f"{'':6s}{'call':>11s}{'put':>11s}{'call bump':>11s}")
pc = pricers[PutCallFwd.CALL]
hs, h = 0.01, 1e-4


def bumped(attribute: str, step: float) -> float:
    owner = pc if attribute == 'valuation_time' else model
    base = getattr(owner, attribute)
    setattr(owner, attribute, base + step)
    value = pc.calc_fair_value()
    setattr(owner, attribute, base)
    return value


bumps = {'delta': (bumped('spot', hs) - bumped('spot', -hs)) / (2 * hs),
         'gamma': (bumped('spot', hs) - 2 * call + bumped('spot', -hs)) / hs**2,
         'vega': (bumped('vol', h) - bumped('vol', -h)) / (2 * h),
         'theta': (bumped('valuation_time', h) - bumped('valuation_time', -h)) / (2 * h),
         'rho': (bumped('risk_free_rate', h) - bumped('risk_free_rate', -h)) / (2 * h)}
for g, b in bumps.items():
    gc = getattr(pricers[PutCallFwd.CALL], f'calc_{g}')()
    gp = getattr(pricers[PutCallFwd.PUT], f'calc_{g}')()
    print(f'{g:6s}{gc:11.6f}{gp:11.6f}{b:11.6f}')
