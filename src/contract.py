from __future__ import annotations
from abc import ABC, abstractmethod
from enums import *


class Contract(ABC):
    def __init__(self, und: Stock, dtype: PutCallFwd, strk: float, exp: float) -> None:
        self._underlying: Stock = und
        self._derivative_type: PutCallFwd = dtype
        self._strike: float = strk
        self._expiry: float = exp

    def get_und(self) -> Stock:
        return self._underlying

    def get_type(self) -> PutCallFwd:
        return self._derivative_type

    def get_strike(self) -> float:
        return self._strike

    def get_expiry(self) -> float:
        return self._expiry

    def __str__(self) -> str:
        return str(self.to_dict())

    @abstractmethod
    def to_dict(self) -> dict[str, any]:
        return {
            "underlying": self._underlying.value,
            "type": self._derivative_type.value,
            "strike": self._strike,
            "expiry": self._expiry
        }

    @abstractmethod
    def convert_to_generic(self) -> GenericContract:
        pass

    @abstractmethod
    def get_timeline(self) -> list[float]:
        pass

    def _raise_incorrect_derivative_type(self):
        raise TypeError(f'Derivative type of {type(self).__name__} must be CALL or PUT')


class VanillaContract(Contract):
    def to_dict(self) -> dict[str, any]:
        return super().to_dict()

    @abstractmethod
    def convert_to_generic(self) -> GenericContract:
        pass

    @abstractmethod
    def payoff(self, spot: float) -> float:
        pass

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


class VanillaOptionContract(VanillaContract):
    def __init__(self, und: Stock, dtype: PutCallFwd, strk: float, exp: float) -> None:
        if dtype not in [PutCallFwd.CALL, PutCallFwd.PUT]:
            self._raise_incorrect_derivative_type()
        super().__init__(und, dtype, strk, exp)

    @abstractmethod
    def convert_to_generic(self) -> GenericContract:
        pass

    @abstractmethod
    def payoff(self, spot: float) -> float:
        pass


class EuropeanContract(VanillaOptionContract):
    def convert_to_generic(self) -> GenericContract:
        # todo: needs to be properly initialized after GenericContract is implemented
        return GenericContract()

    def payoff(self, spot: float) -> float:
        if self._derivative_type == PutCallFwd.CALL:
            return max(spot - self._strike, 0)
        elif self._derivative_type == PutCallFwd.PUT:
            return max(self._strike - spot, 0)
        else:
            self._raise_incorrect_derivative_type()


class AmericanContract(VanillaOptionContract):
    def convert_to_generic(self) -> GenericContract:
        # todo: needs to be properly initialized after GenericContract is implemented
        return GenericContract()

    def payoff(self, spot: float) -> float:
        if self._derivative_type == PutCallFwd.CALL:
            return max(spot - self._strike, 0)
        elif self._derivative_type == PutCallFwd.PUT:
            return max(self._strike - spot, 0)
        else:
            self._raise_incorrect_derivative_type()


class EuropeanDigitalContract(VanillaOptionContract):
    def convert_to_generic(self) -> GenericContract:
        # todo: needs to be properly initialized after GenericContract is implemented
        return GenericContract()

    def payoff(self, spot: float) -> float:
        if self._derivative_type == PutCallFwd.CALL:
            return float(spot - self._strike > 0)
        elif self._derivative_type == PutCallFwd.PUT:
            return float(self._strike - spot > 0)
        else:
            self._raise_incorrect_derivative_type()


class GenericContract:
    # todo: needs to be implemented, defined temporarily as empty class due to annotations
    pass
