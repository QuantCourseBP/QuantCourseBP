from __future__ import annotations
from abc import ABC, abstractmethod
from enums import *


class Contract(ABC):
    timeline_digits: int = 6

    def __init__(self, underlying: Stock, derivative_type: PutCallFwd, long_short: LongShort, strike: float,
                 expiry: float) -> None:
        self.underlying: Stock = underlying
        self.derivative_type: PutCallFwd = derivative_type
        self.long_short: LongShort = long_short
        self.direction: float = 1.0 if self.long_short == LongShort.LONG else -1.0
        self.strike: float = strike
        self.expiry: float = expiry

    def __str__(self) -> str:
        return str(self.to_dict())

    def to_dict(self) -> dict[str, any]:
        return {
            'underlying': self.underlying,
            'type': self.derivative_type,
            'long_short': self.long_short,
            'strike': self.strike,
            'expiry': self.expiry
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
