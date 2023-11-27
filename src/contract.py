from __future__ import annotations
from abc import ABC, abstractmethod
from src.enums import *
from src.utils import *
from statistics import mean
import numpy as np


class Contract(ABC):
    timeline_digits: int = 6

    def __init__(self, underlying: Stock, derivative_type: PutCallFwd, long_short: LongShort, strike: float,
                 expiry: float, num_mon: int = 1) -> None:
        self.underlying: Stock = underlying
        self.derivative_type: PutCallFwd = derivative_type
        self.long_short: LongShort = long_short
        self.direction: float = 1.0 if self.long_short == LongShort.LONG else -1.0
        self.strike: float = strike
        self.expiry: float = expiry
        self.num_mon: int = round(max(num_mon, 1))  # Asian: nr of averaging points; Barrier: nr of monitoring points
        self.contract_type: str = type(self).get_contract_type()

    @classmethod
    def get_contract_type(cls):
        name = cls.__name__
        return name if name == Contract.__name__ else name.removesuffix('Contract')

    def __str__(self) -> str:
        return str(self.to_dict())

    def to_dict(self) -> dict[str, any]:
        return {
            'contract': self.contract_type,
            'underlying': self.underlying,
            'type': self.derivative_type,
            'long_short': self.long_short,
            'strike': self.strike,
            'expiry': self.expiry,
            'observations': self.num_mon
        }

    @abstractmethod
    def get_timeline(self) -> list[float]:
        pass

    @abstractmethod
    def payoff(self, spot: dict[float, float]) -> float:
        pass

    def raise_incorrect_derivative_type_error(
            self,
            supported: tuple[PutCallFwd, ...] = (PutCallFwd.CALL, PutCallFwd.PUT)) -> None:
        raise ValueError(f'Derivative type of {type(self).__name__} must be one of '
                         f'{", ".join(supported)}, but received {self.derivative_type}')

    def raise_missing_spot_error(self, received: list[float]):
        raise ValueError(f'{type(self).__name__} expects spot price on timeline {self.get_timeline()}, '
                         f'but received on {received}')


class ForwardContract(Contract):
    def __init__(self, underlying: Stock, long_short: LongShort, strike: float, expiry: float) -> None:
        super().__init__(underlying, PutCallFwd.FWD, long_short, strike, expiry)

    def get_timeline(self) -> list[float]:
        return [round(self.expiry, self.timeline_digits)]

    def payoff(self, spot: dict[float, float]) -> float:
        t = self.get_timeline()[0]
        if t not in spot.keys():
            self.raise_missing_spot_error(list(spot.keys()))
        return self.direction * (spot[t] - self.strike)


class EuropeanContract(Contract):
    def __init__(self, underlying: Stock, derivative_type: PutCallFwd, long_short: LongShort, strike: float,
                 expiry: float) -> None:
        if derivative_type not in [PutCallFwd.CALL, PutCallFwd.PUT]:
            self.raise_incorrect_derivative_type_error()
        super().__init__(underlying, derivative_type, long_short, strike, expiry)

    def get_timeline(self) -> list[float]:
        return [round(self.expiry, self.timeline_digits)]

    def payoff(self, spot: dict[float, float]) -> float:
        t = self.get_timeline()[0]
        if t not in spot.keys():
            self.raise_missing_spot_error(list(spot.keys()))
        if self.derivative_type == PutCallFwd.CALL:
            return self.direction * max(spot[t] - self.strike, 0)
        elif self.derivative_type == PutCallFwd.PUT:
            return self.direction * max(self.strike - spot[t], 0)
        else:
            self.raise_incorrect_derivative_type_error()


class AmericanContract(Contract):
    def __init__(self, underlying: Stock, derivative_type: PutCallFwd, long_short: LongShort, strike: float,
                 expiry: float) -> None:
        if derivative_type not in [PutCallFwd.CALL, PutCallFwd.PUT]:
            self.raise_incorrect_derivative_type_error()
        super().__init__(underlying, derivative_type, long_short, strike, expiry)

    def get_timeline(self) -> list[float]:
        return [round(self.expiry, self.timeline_digits)]

    def payoff(self, spot: dict[float, float]) -> float:
        t = self.get_timeline()[0]
        if t not in spot.keys():
            self.raise_missing_spot_error(list(spot.keys()))
        if self.derivative_type == PutCallFwd.CALL:
            return self.direction * max(spot[t] - self.strike, 0)
        elif self.derivative_type == PutCallFwd.PUT:
            return self.direction * max(self.strike - spot[t], 0)
        else:
            self.raise_incorrect_derivative_type_error()


class EuropeanDigitalContract(Contract):
    def __init__(self, underlying: Stock, derivative_type: PutCallFwd, long_short: LongShort, strike: float,
                 expiry: float) -> None:
        if derivative_type not in [PutCallFwd.CALL, PutCallFwd.PUT]:
            self.raise_incorrect_derivative_type_error()
        super().__init__(underlying, derivative_type, long_short, strike, expiry)

    def get_timeline(self) -> list[float]:
        return [round(self.expiry, self.timeline_digits)]

    def payoff(self, spot: dict[float, float]) -> float:
        t = self.get_timeline()[0]
        if t not in spot.keys():
            self.raise_missing_spot_error(list(spot.keys()))
        if self.derivative_type == PutCallFwd.CALL:
            return self.direction * float(spot[t] - self.strike > 0)
        elif self.derivative_type == PutCallFwd.PUT:
            return self.direction * float(self.strike - spot[t] > 0)
        else:
            self.raise_incorrect_derivative_type_error()


class AsianContract(Contract):
    def __init__(self, underlying: Stock, derivative_type: PutCallFwd, long_short: LongShort, strike: float,
                 expiry: float, num_mon: int) -> None:
        if derivative_type not in [PutCallFwd.CALL, PutCallFwd.PUT]:
            self.raise_incorrect_derivative_type_error()
        super().__init__(underlying, derivative_type, long_short, strike, expiry, num_mon)

    def get_timeline(self) -> list[float]:
        return [round(((i+1) / self.num_mon) * self.expiry, self.timeline_digits) for i in range(self.num_mon)]

    def payoff(self, spot: dict[float, float]) -> float:
        timeline = self.get_timeline()
        if not set(timeline).issubset(set(spot.keys())):
            self.raise_missing_spot_error(list(spot.keys()))
        obs = [spot[t] for t in timeline]
        if self.derivative_type == PutCallFwd.CALL:
            return self.direction * max(mean(obs) - self.strike, 0)
        elif self.derivative_type == PutCallFwd.PUT:
            return self.direction * max(self.strike - mean(obs), 0)
        else:
            self.raise_incorrect_derivative_type_error()


class EuropeanBarrierContract(Contract):
    def __init__(self, underlying: Stock, derivative_type: PutCallFwd, long_short: LongShort, strike: float,
                 expiry: float, num_mon: int, barrier: float, up_down: UpDown, in_out: InOut) -> None:
        if derivative_type not in [PutCallFwd.CALL, PutCallFwd.PUT]:
            self.raise_incorrect_derivative_type_error()
        super().__init__(underlying, derivative_type, long_short, strike, expiry, num_mon)
        self.barrier: Barrier = Barrier(barrier, up_down, in_out)

    def to_dict(self) -> dict[str, any]:
        out = super().to_dict()
        out |= {
            'barrier': self.barrier.barrier_level,
            'up_down': self.barrier.up_down,
            'in_out': self.barrier.in_out
        }
        return out

    def get_timeline(self) -> list[float]:
        return [round(((i+1) / self.num_mon) * self.expiry, self.timeline_digits) for i in range(self.num_mon)]

    def payoff(self, spot: dict[float, float], vol: float = np.nan) -> float:
        timeline = self.get_timeline()
        if not set(timeline).issubset(set(spot.keys())):
            self.raise_missing_spot_error(list(spot.keys()))
        obs = [spot[t] for t in timeline]
        in_out = self.barrier.in_out
        is_breached = self.barrier.is_breached(spot, vol)

        mult = float(int(in_out == InOut.IN) * is_breached + int(in_out == InOut.OUT) * (1 - is_breached))
        if self.derivative_type == PutCallFwd.CALL:
            return self.direction * mult * max(obs[-1] - self.strike, 0)
        elif self.derivative_type == PutCallFwd.PUT:
            return self.direction * mult * max(self.strike - obs[-1], 0)
        else:
            self.raise_incorrect_derivative_type_error()


class Barrier:
    def __init__(self, barrier_level: float, up_down: UpDown, in_out: InOut) -> None:
        self.barrier_level: float = barrier_level
        if up_down not in [UpDown.UP, UpDown.DOWN]:
            self.raise_incorrect_up_down_type()
        self.up_down: UpDown = up_down
        if in_out not in [InOut.IN, in_out.OUT]:
            self.raise_incorrect_in_out_type()
        self.in_out: InOut = in_out

    def is_breached(self, spot: dict[float, float], vol: float) -> float:
        timeline = list(spot.keys())
        observations = [spot[t] for t in timeline]
        if np.isnan(vol):      # standard case without probabilities
            if self.up_down == UpDown.UP:
                return float(any([self.barrier_level <= price for price in observations]))
            elif self.up_down == UpDown.DOWN:
                return float(any([self.barrier_level >= price for price in observations]))
            else:
                self.raise_incorrect_up_down_type()
        else:
            probs_no_breach = []
            for i in range(len(timeline)-1):
                prob = prob_breach_barrier_segment(self.barrier_level, vol, timeline[i], timeline[i+1],
                                                   observations[i], observations[i+1], self.up_down)
                probs_no_breach.append(1-prob)
            return 1 - np.prod(probs_no_breach)


    def raise_incorrect_up_down_type(self):
        raise TypeError(f'Updown parameter of {type(self).__name__} must be UP or DOWN')

    def raise_incorrect_in_out_type(self):
        raise TypeError(f'Inout parameter of {type(self).__name__} must be IN or OUT')
