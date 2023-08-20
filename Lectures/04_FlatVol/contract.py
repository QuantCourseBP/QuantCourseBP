from __future__ import annotations
from abc import ABC, abstractmethod
from src.enums import *


class Contract(ABC):
    TIMELINE_DIGITS: int = 6

    def __init__(self, und: Stock, dtype: PutCallFwd, long_short: LongShort,
                 strk: float, exp: float) -> None:
        self._underlying: Stock = und
        self._derivative_type: PutCallFwd = dtype
        self._long_short: LongShort = long_short
        self._direction: float = 1.0 if self._long_short == LongShort.LONG else -1.0
        self._strike: float = strk
        self._expiry: float = exp

    def get_underlying(self) -> Stock:
        return self._underlying

    def get_type(self) -> PutCallFwd:
        return self._derivative_type

    def get_long_short(self) -> LongShort:
        return self._long_short

    def get_direction(self) -> float:
        return self._direction

    def get_strike(self) -> float:
        return self._strike

    def get_expiry(self) -> float:
        return self._expiry

    def __str__(self) -> str:
        return str(self.to_dict())

    def to_dict(self) -> dict[str, any]:
        return {
            'underlying': self._underlying,
            'type': self._derivative_type,
            'long_short': self._long_short,
            'strike': self._strike,
            'expiry': self._expiry,
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
                         f'{", ".join(supported)}, but received {self.get_type()}')

    def _raise_missing_spot_error(self, received: list[float]):
        raise ValueError(f'{type(self).__name__} expects spot price on timeline {self.get_timeline()}, '
                         f'but received on {received}')


class ForwardContract(Contract):
    def __init__(self, und: Stock, long_short: LongShort, strk: float, exp: float) -> None:
        super().__init__(und, PutCallFwd.FWD, long_short, strk, exp)

    def get_timeline(self) -> list[float]:
        return [round(self._expiry, self.TIMELINE_DIGITS)]

    def payoff(self, spot: dict[float, float]) -> float:
        t = self.get_timeline()[0]
        if t not in spot.keys():
            self._raise_missing_spot_error(list(spot.keys()))
        return self._direction * (spot[t] - self._strike)


class EuropeanContract(Contract):
    def __init__(self, und: Stock, dtype: PutCallFwd, long_short: LongShort, strk: float, exp: float) -> None:
        if dtype not in [PutCallFwd.CALL, PutCallFwd.PUT]:
            self.raise_incorrect_derivative_type_error()
        super().__init__(und, dtype, long_short, strk, exp)

    def get_timeline(self) -> list[float]:
        return [round(self._expiry, self.TIMELINE_DIGITS)]

    def payoff(self, spot: dict[float, float]) -> float:
        t = self.get_timeline()[0]
        if t not in spot.keys():
            self._raise_missing_spot_error(list(spot.keys()))
        if self._derivative_type == PutCallFwd.CALL:
            return self._direction * max(spot[t] - self._strike, 0)
        elif self._derivative_type == PutCallFwd.PUT:
            return self._direction * max(self._strike - spot[t], 0)
        else:
            self.raise_incorrect_derivative_type_error()
