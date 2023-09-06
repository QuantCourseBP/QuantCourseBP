from __future__ import annotations
from abc import ABC, abstractmethod
from src.enums import *
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
                         f'{", ".join(supported)}, but received {self.derivative_type}')

    def raise_missing_spot_error(self, received: list[float]):
        raise ValueError(f'{type(self).__name__} expects spot price on timeline {self.get_timeline()}, '
                         f'but received on {received}')


class ForwardContract(Contract):
    def __init__(self, underlying: Stock, long_short: LongShort, strike: float, expiry: float) -> None:
        super().__init__(underlying, PutCallFwd.FWD, long_short, strike, expiry)

    def convert_to_generic(self) -> GenericContract:
        return GenericContract(self.contract_type, self.underlying, self.derivative_type, self.long_short,
                               self.strike, self.expiry)

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

    def convert_to_generic(self) -> GenericContract:
        return GenericContract(self.contract_type, self.underlying, self.derivative_type, self.long_short,
                               self.strike, self.expiry)

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

    def convert_to_generic(self) -> GenericContract:
        return GenericContract(self.contract_type, self.underlying, self.derivative_type, self.long_short,
                               self.strike, self.expiry)

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

    def convert_to_generic(self) -> GenericContract:
        return GenericContract(self.contract_type, self.underlying, self.derivative_type, self.long_short,
                               self.strike, self.expiry)

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

    def convert_to_generic(self) -> GenericContract:
        return GenericContract(self.contract_type, self.underlying, self.derivative_type, self.long_short,
                               self.strike, self.expiry, self.num_mon)

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

    def convert_to_generic(self) -> GenericContract:
        return GenericContract(self.contract_type, self.underlying, self.derivative_type, self.long_short,
                               self.strike, self.expiry, self.num_mon, self.barrier.barrier_level,
                               self.barrier.up_down, self.barrier.in_out)

    def get_timeline(self) -> list[float]:
        return [round(((i+1) / self.num_mon) * self.expiry, self.timeline_digits) for i in range(self.num_mon)]

    def payoff(self, spot: dict[float, float]) -> float:
        timeline = self.get_timeline()
        if not set(timeline).issubset(set(spot.keys())):
            self.raise_missing_spot_error(list(spot.keys()))
        obs = [spot[t] for t in timeline]
        in_out = self.barrier.in_out
        is_breached = int(self.barrier.is_breached(obs))
        mult = float(int(in_out == InOut.IN) * is_breached + int(in_out == InOut.OUT) * (1 - is_breached))
        if self.derivative_type == PutCallFwd.CALL:
            return self.direction * mult * max(obs[-1] - self.strike, 0)
        elif self.derivative_type == PutCallFwd.PUT:
            return self.direction * mult * max(self.strike - obs[-1], 0)
        else:
            self.raise_incorrect_derivative_type_error()


class GenericContract(Contract):
    def __init__(self, contract_type: str, underlying: Stock, derivative_type: PutCallFwd, long_short: LongShort, strike: float,
                 expiry: float, num_mon: int = 1, barrier: float = np.nan, up_down: UpDown | None = None,
                 in_out: InOut | None = None) -> None:
        supported_deriv_types = (PutCallFwd.CALL, PutCallFwd.PUT, PutCallFwd.FWD)
        if derivative_type not in supported_deriv_types:
            self.raise_incorrect_derivative_type_error(supported_deriv_types)
        super().__init__(underlying, derivative_type, long_short, strike, expiry, num_mon)
        self.contract_type: str = contract_type
        self.barrier: Barrier | None = None
        if barrier != np.nan and up_down is not None and in_out is not None:
            self.barrier = Barrier(barrier, up_down, in_out)

    def to_dict(self) -> dict[str, any]:
        out = super().to_dict()
        if self.barrier is not None:
            out |= {
                'barrier': self.barrier.barrier_level,
                'up_down': self.barrier.up_down,
                'in_out': self.barrier.in_out
            }
        return out

    def convert_to_generic(self) -> GenericContract:
        return self

    def get_timeline(self) -> list[float]:
        return [round(((i+1) / self.num_mon) * self.expiry, self.timeline_digits) for i in range(self.num_mon)]

    def payoff(self, spot: dict[float, float]) -> float:
        timeline = self.get_timeline()
        if not set(timeline).issubset(set(spot.keys())):
            self.raise_missing_spot_error(list(spot.keys()))
        obs = [spot[t] for t in timeline]
        payoff = mean(obs) - self.strike
        call_put = 1.0 if self.derivative_type == PutCallFwd.CALL else -1.0
        if self.contract_type == ForwardContract.get_contract_type():
            return self.direction * payoff
        elif self.contract_type == EuropeanDigitalContract.get_contract_type():
            return self.direction * float(call_put * payoff > 0)
        elif self.contract_type == EuropeanBarrierContract.get_contract_type():
            mult = 1.0
            if self.barrier is not None:
                in_out = self.barrier.in_out
                is_breached = int(self.barrier.is_breached(obs))
                mult = float(int(in_out == InOut.IN) * is_breached + int(in_out == InOut.OUT) * (1 - is_breached))
            payoff = (obs[-1] - self.strike)
            return self.direction * mult * max(call_put * payoff, 0)
        else:
            return self.direction * max(call_put * payoff, 0)


class Barrier:
    def __init__(self, barrier_level: float, up_down: UpDown, in_out: InOut) -> None:
        self.barrier_level: float = barrier_level
        if up_down not in [UpDown.UP, UpDown.DOWN]:
            self.raise_incorrect_up_down_type()
        self.up_down: UpDown = up_down
        if in_out not in [InOut.IN, in_out.OUT]:
            self.raise_incorrect_in_out_type()
        self.in_out: InOut = in_out

    def in_out(self) -> InOut:
        return self.in_out

    def is_breached(self, observations: list[float]) -> bool:
        if self.up_down == UpDown.UP:
            return any([self.barrier_level <= price for price in observations])
        elif self.up_down == UpDown.DOWN:
            return any([self.barrier_level >= price for price in observations])
        else:
            self.raise_incorrect_up_down_type()

    def raise_incorrect_up_down_type(self):
        raise TypeError(f'Updown parameter of {type(self).__name__} must be UP or DOWN')

    def raise_incorrect_in_out_type(self):
        raise TypeError(f'Inout parameter of {type(self).__name__} must be IN or OUT')
