from __future__ import annotations
from abc import ABC, abstractmethod
from src.enums import *
from statistics import mean


class Contract(ABC):
    def __init__(self, und: Stock, dtype: PutCallFwd, strk: float, exp: float, num_mon: int = 1) -> None:
        self._underlying: Stock = und
        self._derivative_type: PutCallFwd = dtype
        self._strike: float = strk
        self._expiry: float = exp
        self._num_mon: int = num_mon    # Asian: nr of averaging points; Barrier: nr of monitoring points

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

    # @abstractmethod
    # def convert_to_generic(self) -> GenericContract:
    #     pass

    @abstractmethod
    def get_timeline(self) -> list[float]:
        pass

    def _raise_incorrect_derivative_type(self):
        raise TypeError(f'Derivative type of {type(self).__name__} must be CALL or PUT')


class VanillaContract(Contract):
    def to_dict(self) -> dict[str, any]:
        return super().to_dict()

    # @abstractmethod
    # def convert_to_generic(self) -> GenericContract:
    #     pass

    @abstractmethod
    def payoff(self, spot: float) -> float:
        pass

    def get_timeline(self) -> list[float]:
        return [self._expiry]


class ForwardContract(VanillaContract):
    def __init__(self, und: Stock, strk: float, exp: float) -> None:
        super().__init__(und, PutCallFwd.FWD, strk, exp)

    # def convert_to_generic(self) -> GenericContract:
    #     # todo: needs to be properly initialized after GenericContract is implemented
    #     return GenericContract()

    def payoff(self, spot: float) -> float:
        return spot - self._strike


class VanillaOptionContract(VanillaContract):
    def __init__(self, und: Stock, dtype: PutCallFwd, strk: float, exp: float) -> None:
        if dtype not in [PutCallFwd.CALL, PutCallFwd.PUT]:
            self._raise_incorrect_derivative_type()
        super().__init__(und, dtype, strk, exp)

    # @abstractmethod
    # def convert_to_generic(self) -> GenericContract:
    #     pass

    @abstractmethod
    def payoff(self, spot: float) -> float:
        pass


class EuropeanContract(VanillaOptionContract):
    # def convert_to_generic(self) -> GenericContract:
    #     # todo: needs to be properly initialized after GenericContract is implemented
    #     return GenericContract()

    def payoff(self, spot: float) -> float:
        if self._derivative_type == PutCallFwd.CALL:
            return max(spot - self._strike, 0)
        elif self._derivative_type == PutCallFwd.PUT:
            return max(self._strike - spot, 0)
        else:
            self._raise_incorrect_derivative_type()


class AmericanContract(VanillaOptionContract):
    # def convert_to_generic(self) -> GenericContract:
    #     # todo: needs to be properly initialized after GenericContract is implemented
    #     return GenericContract()

    def payoff(self, spot: float) -> float:
        if self._derivative_type == PutCallFwd.CALL:
            return max(spot - self._strike, 0)
        elif self._derivative_type == PutCallFwd.PUT:
            return max(self._strike - spot, 0)
        else:
            self._raise_incorrect_derivative_type()


class EuropeanDigitalContract(VanillaOptionContract):
    # def convert_to_generic(self) -> GenericContract:
    #     # todo: needs to be properly initialized after GenericContract is implemented
    #     return GenericContract()

    def payoff(self, spot: float) -> float:
        if self._derivative_type == PutCallFwd.CALL:
            return float(spot - self._strike > 0)
        elif self._derivative_type == PutCallFwd.PUT:
            return float(self._strike - spot > 0)
        else:
            self._raise_incorrect_derivative_type()


class PathDependentContract(Contract):

    def to_dict(self) -> dict[str, any]:
        return super().to_dict()

    # @abstractmethod
    # def convert_to_generic(self) -> GenericContract:
    #     pass

    @abstractmethod
    def payoff(self, spot: float) -> float:
        pass

    def get_timeline(self) -> list[float]:
        return [((i+1)/self._num_mon)*self._expiry for i in range(self._num_mon)]


class AsianContract(PathDependentContract):

    # def convert_to_generic(self) -> GenericContract:
    #     # todo: needs to be properly initialized after GenericContract is implemented
    #     return GenericContract()

    def payoff(self, prices_und: float) -> float:
    # TO DO: prices_und to derive from the underlying process using the timeline
        if self._derivative_type == PutCallFwd.CALL:
            return max(mean(prices_und) - self._strike, 0)
        elif self._derivative_type == PutCallFwd.PUT:
            return max(self._strike - mean(prices_und), 0)
        else:
            self._raise_incorrect_derivative_type()


class EuropeanBarrierContract(PathDependentContract):

    # def convert_to_generic(self) -> GenericContract:
    #     # todo: needs to be properly initialized after GenericContract is implemented
    #     return GenericContract()

    def payoff(self, spot: float) -> float:
        if self._derivative_type == PutCallFwd.CALL:
            return max(spot - self._strike, 0)
        elif self._derivative_type == PutCallFwd.PUT:
            return max(self._strike - spot, 0)
        else:
            self._raise_incorrect_derivative_type()


class GenericContract(PathDependentContract):
    # todo: needs to be implemented, defined temporarily as empty class due to annotations
    pass

    def payoff(self, spot: float) -> float:
        if self._derivative_type == PutCallFwd.CALL:
            return max(spot - self._strike, 0)
        elif self._derivative_type == PutCallFwd.PUT:
            return max(self._strike - spot, 0)
        else:
            self._raise_incorrect_derivative_type()


def main():

    trade1 = ForwardContract('Apple', 1, 2)
    print(trade1.get_und())
    print(trade1.get_type())
    print("Strike: {}".format(trade1.get_strike()))
    print(trade1.payoff(0.5))
    print(trade1.payoff(1.5))

    print("\n")
    trade2 = AsianContract('Microsoft', 'CALL', 1, 2)
    print(trade2.get_und())
    print(trade2.get_type())
    print("Strike: {}".format(trade2.get_strike()))
    print(trade2.payoff([1.8, 1.9, 2, 1.9, 2.1]))
    print(trade2.payoff([0.8, 0.9, 1, 0.9, 1.1]))


if __name__ == '__main__': main()

