from __future__ import annotations
from abc import ABC, abstractmethod
from enums import *
import numpy as np


class Contract(ABC):
    _underlying: Stock
    _derivative_type: PutCallFwd
    _strike: float
    _expiry: float

    def get_und(self) -> Stock:
        return self._underlying

    def get_type(self) -> PutCallFwd:
        return self._derivative_type

    def get_strike(self) -> float:
        return self._strike

    def get_expiry(self) -> float:
        return self._expiry

    @abstractmethod
    def display(self) -> dict[str, any]:
        pass

    @abstractmethod
    def convert_to_generic(self) -> GenericContract:
        pass

    @abstractmethod
    def get_timeline(self) -> list[float]:
        pass


class VanillaContract(Contract):
    @abstractmethod
    def convert_to_generic(self) -> GenericContract:
        pass

    @abstractmethod
    def payoff(self, spot: float) -> float:
        pass

    def __init__(self, und: Stock, dtype: PutCallFwd, strk: float, exp: float) -> None:
        self._underlying = und
        self._derivative_type = dtype
        self._strike = strk
        self._expiry = exp

    def display(self) -> dict[str, any]:
        return {
            "underlying": self._underlying.value,
            "type": self._derivative_type.value,
            "strike": self._strike,
            "expiry": self._expiry
        }

    def get_timeline(self) -> list[float]:
        return [self._expiry]


class ForwardContract(VanillaContract):
    def __init__(self, und: Stock, strk: float, exp: float) -> None:
        super().__init__(und, PutCallFwd.FWD, strk, exp)

    def convert_to_generic(self) -> GenericContract:
        # todo: needs to be properly initialized after GenericContract is implemented
        return GenericContract()

    def payoff(self, spot: float) -> float:
        return spot - self._strike


class EuropeanContract(VanillaContract):
    def __init__(self, und: Stock, dtype: PutCallFwd, strk: float, exp: float) -> None:
        if dtype not in [PutCallFwd.CALL, PutCallFwd.PUT]:
            raise TypeError(f'Derivative type of {type(self).__name__} must be CALL or PUT')
        super().__init__(und, dtype, strk, exp)

    def convert_to_generic(self) -> GenericContract:
        # todo: needs to be properly initialized after GenericContract is implemented
        return GenericContract()

    def payoff(self, spot: float) -> float:
        if self._derivative_type == PutCallFwd.CALL:
            return max(spot - self._strike, 0)
        elif self._derivative_type == PutCallFwd.PUT:
            return max(self._strike - spot, 0)
        else:
            return np.NaN


class AmericanContract(VanillaContract):
    def __init__(self, und: Stock, dtype: PutCallFwd, strk: float, exp: float) -> None:
        if dtype not in [PutCallFwd.CALL, PutCallFwd.PUT]:
            raise TypeError(f'Derivative type of {type(self).__name__} must be CALL or PUT')
        super().__init__(und, dtype, strk, exp)

    def convert_to_generic(self) -> GenericContract:
        # todo: needs to be properly initialized after GenericContract is implemented
        return GenericContract()

    def payoff(self, spot: float) -> float:
        if self._derivative_type == PutCallFwd.CALL:
            return max(spot - self._strike, 0)
        elif self._derivative_type == PutCallFwd.PUT:
            return max(self._strike - spot, 0)
        else:
            return np.NaN


class EuropeanDigitalContract(VanillaContract):
    def __init__(self, und: Stock, dtype: PutCallFwd, strk: float, exp: float) -> None:
        if dtype not in [PutCallFwd.CALL, PutCallFwd.PUT]:
            raise TypeError(f'Derivative type of {type(self).__name__} must be CALL or PUT')
        super().__init__(und, dtype, strk, exp)

    def convert_to_generic(self) -> GenericContract:
        # todo: needs to be properly initialized after GenericContract is implemented
        return GenericContract()

    def payoff(self, spot: float) -> float:
        if self._derivative_type == PutCallFwd.CALL:
            return float(spot - self._strike > 0)
        elif self._derivative_type == PutCallFwd.PUT:
            return float(self._strike - spot > 0)
        else:
            return np.NaN


class GenericContract:
    # todo: needs to be implemented, defined temporarily as empty class due to annotations
    pass
