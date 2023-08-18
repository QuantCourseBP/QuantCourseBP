from __future__ import annotations
from abc import ABC, abstractmethod
from src.enums import *
from statistics import mean
import numpy as np


class Contract(ABC):
    TIMELINE_DIGITS: int = 6

    def __init__(self, und: Stock, dtype: PutCallFwd, long_short: LongShort,
                 strk: float, exp: float, num_mon: int = 1) -> None:
        self._underlying: Stock = und
        self._derivative_type: PutCallFwd = dtype
        self._long_short: LongShort = long_short
        self._direction: float = 1.0 if self._long_short == LongShort.LONG else -1.0
        self._strike: float = strk
        self._expiry: float = exp
        self._num_mon: int = round(max(num_mon, 1))  # Asian: nr of averaging points; Barrier: nr of monitoring points
        self._contract_type: str = type(self).get_contract_type_static()

    @classmethod
    def get_contract_type_static(cls):
        name = cls.__name__
        return name if name == Contract.__name__ else name.removesuffix('Contract')

    def get_contract_type(self) -> str:
        return self._contract_type

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
            'contract': self._contract_type,
            'underlying': self._underlying,
            'type': self._derivative_type,
            'long_short': self._long_short,
            'strike': self._strike,
            'expiry': self._expiry,
            'observations': self._num_mon
        }

    @abstractmethod
    def convert_to_generic(self) -> GenericContract:
        pass

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

    def convert_to_generic(self) -> GenericContract:
        return GenericContract(self._contract_type, self._underlying, self._derivative_type, self._long_short,
                               self._strike, self._expiry)

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

    def convert_to_generic(self) -> GenericContract:
        return GenericContract(self._contract_type, self._underlying, self._derivative_type, self._long_short,
                               self._strike, self._expiry)

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


class AmericanContract(Contract):
    def __init__(self, und: Stock, dtype: PutCallFwd, long_short: LongShort, strk: float, exp: float) -> None:
        if dtype not in [PutCallFwd.CALL, PutCallFwd.PUT]:
            self.raise_incorrect_derivative_type_error()
        super().__init__(und, dtype, long_short, strk, exp)

    def convert_to_generic(self) -> GenericContract:
        return GenericContract(self._contract_type, self._underlying, self._derivative_type, self._long_short,
                               self._strike, self._expiry)

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


class EuropeanDigitalContract(Contract):
    def __init__(self, und: Stock, dtype: PutCallFwd, long_short: LongShort, strk: float, exp: float) -> None:
        if dtype not in [PutCallFwd.CALL, PutCallFwd.PUT]:
            self.raise_incorrect_derivative_type_error()
        super().__init__(und, dtype, long_short, strk, exp)

    def convert_to_generic(self) -> GenericContract:
        return GenericContract(self._contract_type, self._underlying, self._derivative_type, self._long_short,
                               self._strike, self._expiry)

    def get_timeline(self) -> list[float]:
        return [round(self._expiry, self.TIMELINE_DIGITS)]

    def payoff(self, spot: dict[float, float]) -> float:
        t = self.get_timeline()[0]
        if t not in spot.keys():
            self._raise_missing_spot_error(list(spot.keys()))
        if self._derivative_type == PutCallFwd.CALL:
            return self._direction * float(spot[t] - self._strike > 0)
        elif self._derivative_type == PutCallFwd.PUT:
            return self._direction * float(self._strike - spot[t] > 0)
        else:
            self.raise_incorrect_derivative_type_error()


class AsianContract(Contract):
    def __init__(self, und: Stock, dtype: PutCallFwd, long_short: LongShort, strk: float, exp: float,
                 num_mon: int) -> None:
        if dtype not in [PutCallFwd.CALL, PutCallFwd.PUT]:
            self.raise_incorrect_derivative_type_error()
        super().__init__(und, dtype, long_short, strk, exp, num_mon)

    def convert_to_generic(self) -> GenericContract:
        return GenericContract(self._contract_type, self._underlying, self._derivative_type, self._long_short,
                               self._strike, self._expiry, self._num_mon)

    def get_timeline(self) -> list[float]:
        return [round(((i+1) / self._num_mon) * self._expiry, self.TIMELINE_DIGITS) for i in range(self._num_mon)]

    def payoff(self, spot: dict[float, float]) -> float:
        timeline = self.get_timeline()
        if not set(timeline).issubset(set(spot.keys())):
            self._raise_missing_spot_error(list(spot.keys()))
        obs = [spot[t] for t in timeline]
        if self._derivative_type == PutCallFwd.CALL:
            return self._direction * max(mean(obs) - self._strike, 0)
        elif self._derivative_type == PutCallFwd.PUT:
            return self._direction * max(self._strike - mean(obs), 0)
        else:
            self.raise_incorrect_derivative_type_error()


class EuropeanBarrierContract(Contract):
    def __init__(self, und: Stock, dtype: PutCallFwd, long_short: LongShort, strk: float, exp: float,
                 num_mon: int, barrier: float, up_down: UpDown, in_out: InOut) -> None:
        if dtype not in [PutCallFwd.CALL, PutCallFwd.PUT]:
            self.raise_incorrect_derivative_type_error()
        super().__init__(und, dtype, long_short, strk, exp, num_mon)
        self._barrier: Barrier = Barrier(barrier, up_down, in_out)

    def get_barrier(self) -> Barrier:
        return self._barrier

    def to_dict(self) -> dict[str, any]:
        out = super().to_dict()
        out |= {
            'barrier': self._barrier.get_barrier_level(),
            'up_down': self._barrier.get_up_down(),
            'in_out': self._barrier.get_in_out()
        }
        return out

    def convert_to_generic(self) -> GenericContract:
        return GenericContract(self._contract_type, self._underlying, self._derivative_type, self._long_short,
                               self._strike, self._expiry, self._num_mon, self._barrier.get_barrier_level(),
                               self._barrier.get_up_down(), self._barrier.get_in_out())

    def get_timeline(self) -> list[float]:
        return [round(((i+1) / self._num_mon) * self._expiry, self.TIMELINE_DIGITS) for i in range(self._num_mon)]

    def payoff(self, spot: dict[float, float]) -> float:
        timeline = self.get_timeline()
        if not set(timeline).issubset(set(spot.keys())):
            self._raise_missing_spot_error(list(spot.keys()))
        obs = [spot[t] for t in timeline]
        in_out = self._barrier.get_in_out()
        is_breached = int(self._barrier.is_breached(obs))
        mult = float(int(in_out == InOut.IN) * is_breached + int(in_out == InOut.OUT) * (1 - is_breached))
        if self._derivative_type == PutCallFwd.CALL:
            return self._direction * mult * max(obs[-1] - self._strike, 0)
        elif self._derivative_type == PutCallFwd.PUT:
            return self._direction * mult * max(self._strike - obs[-1], 0)
        else:
            self.raise_incorrect_derivative_type_error()


class GenericContract(Contract):
    def __init__(self, contract_type: str, und: Stock, dtype: PutCallFwd, long_short: LongShort, strk: float,
                 exp: float, num_mon: int = 1, barrier: float = np.nan, up_down: UpDown | None = None,
                 in_out: InOut | None = None) -> None:
        supported_deriv_types = (PutCallFwd.CALL, PutCallFwd.PUT, PutCallFwd.FWD)
        if dtype not in supported_deriv_types:
            self.raise_incorrect_derivative_type_error(supported_deriv_types)
        super().__init__(und, dtype, long_short, strk, exp, num_mon)
        self._contract_type: str = contract_type
        self._barrier: Barrier | None = None
        if barrier != np.nan and up_down is not None and in_out is not None:
            self._barrier = Barrier(barrier, up_down, in_out)

    def get_barrier(self) -> Barrier | None:
        return self._barrier

    def to_dict(self) -> dict[str, any]:
        out = super().to_dict()
        if self._barrier is not None:
            out |= {
                'barrier': self._barrier.get_barrier_level(),
                'up_down': self._barrier.get_up_down(),
                'in_out': self._barrier.get_in_out()
            }
        return out

    def convert_to_generic(self) -> GenericContract:
        return self

    def get_timeline(self) -> list[float]:
        return [round(((i+1) / self._num_mon) * self._expiry, self.TIMELINE_DIGITS) for i in range(self._num_mon)]

    def payoff(self, spot: dict[float, float]) -> float:
        timeline = self.get_timeline()
        if not set(timeline).issubset(set(spot.keys())):
            self._raise_missing_spot_error(list(spot.keys()))
        obs = [spot[t] for t in timeline]
        payoff = mean(obs) - self._strike
        call_put = 1.0 if self._derivative_type == PutCallFwd.CALL else -1.0
        if self._contract_type == ForwardContract.get_contract_type_static():
            return self._direction * payoff
        elif self._contract_type == EuropeanDigitalContract.get_contract_type_static():
            return self._direction * float(call_put * payoff > 0)
        elif self._contract_type == EuropeanBarrierContract.get_contract_type_static():
            mult = 1.0
            if self._barrier is not None:
                in_out = self._barrier.get_in_out()
                is_breached = int(self._barrier.is_breached(obs))
                mult = float(int(in_out == InOut.IN) * is_breached + int(in_out == InOut.OUT) * (1 - is_breached))
            payoff = (obs[-1] - self._strike)
            return self._direction * mult * max(call_put * payoff, 0)
        else:
            return self._direction * max(call_put * payoff, 0)


class Barrier:
    def __init__(self, barrier_level: float, up_down: UpDown, in_out: InOut) -> None:
        self._barrier_level: float = barrier_level
        if up_down not in [UpDown.UP, UpDown.DOWN]:
            self._raise_incorrect_up_down_type()
        self._up_down: UpDown = up_down
        if in_out not in [InOut.IN, in_out.OUT]:
            self._raise_incorrect_in_out_type()
        self._in_out: InOut = in_out

    def get_barrier_level(self) -> float:
        return self._barrier_level

    def get_up_down(self) -> UpDown:
        return self._up_down

    def get_in_out(self) -> InOut:
        return self._in_out

    def is_breached(self, observations: list[float]) -> bool:
        if self._up_down == UpDown.UP:
            return any([self._barrier_level <= price for price in observations])
        elif self._up_down == UpDown.DOWN:
            return any([self._barrier_level >= price for price in observations])
        else:
            self._raise_incorrect_up_down_type()

    def _raise_incorrect_up_down_type(self):
        raise TypeError(f'Updown parameter of {type(self).__name__} must be UP or DOWN')

    def _raise_incorrect_in_out_type(self):
        raise TypeError(f'Inout parameter of {type(self).__name__} must be IN or OUT')
