from __future__ import annotations
from abc import ABC, abstractmethod
from scipy.stats import norm
from src.enums import *
from contract import *
from model import *


# TASK:
# 1. Implement calc_fair_value() method in EuropeanAnalyticPricer.


class Params:
    pass


class Pricer(ABC):
    # Only for theta calculation via bump and revaluation
    valuation_time: float = 0.0

    def __init__(self, contract: Contract, model: MarketModel, params: Params) -> None:
        self.contract: Contract = contract
        self.model: MarketModel = model
        self.params: Params = params

    @abstractmethod
    def calc_fair_value(self) -> float:
        pass

    @abstractmethod
    def calc_delta(self) -> float:
        pass

    @abstractmethod
    def calc_gamma(self) -> float:
        pass

    @abstractmethod
    def calc_vega(self) -> float:
        pass

    @abstractmethod
    def calc_theta(self) -> float:
        pass

    @abstractmethod
    def calc_rho(self) -> float:
        pass


class ForwardAnalyticPricer(Pricer):
    supported_deriv_type: tuple[PutCallFwd, ...] = (PutCallFwd.FWD,)

    def __init__(self, contract: ForwardContract, model: MarketModel, params: Params) -> None:
        if not isinstance(contract, ForwardContract):
            raise TypeError(f'Contract must be of type ForwardContract but received {type(contract).__name__}')
        super().__init__(contract, model, params)

    def calc_fair_value(self) -> float:
        direction = self.contract.direction
        spot = self.model.spot
        strike = self.contract.strike
        time_to_expiry = self.contract.expiry - self.valuation_time
        df = self.model.calc_df(time_to_expiry)
        if self.contract.derivative_type == PutCallFwd.FWD:
            return direction * (spot - strike * df)
        else:
            self.contract.raise_incorrect_derivative_type_error(self.supported_deriv_type)

    def calc_delta(self) -> float:
        return self.contract.direction * 1.0

    def calc_gamma(self) -> float:
        return 0.0

    def calc_vega(self) -> float:
        return 0.0

    def calc_theta(self) -> float:
        greek = 0.0
        strike = self.contract.strike
        time_to_expiry = self.contract.expiry - self.valuation_time
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)
        if self.contract.derivative_type == PutCallFwd.FWD:
            greek = -1.0 * strike * df * rate
        else:
            self.contract.raise_incorrect_derivative_type_error(self.supported_deriv_type)
        return self.contract.direction * greek

    def calc_rho(self) -> float:
        greek = 0.0
        strike = self.contract.strike
        time_to_expiry = self.contract.expiry - self.valuation_time
        df = self.model.calc_df(time_to_expiry)
        if self.contract.derivative_type == PutCallFwd.FWD:
            greek = strike * df * time_to_expiry
        else:
            self.contract.raise_incorrect_derivative_type_error(self.supported_deriv_type)
        return self.contract.direction * greek


class EuropeanAnalyticPricer(Pricer):
    @staticmethod
    def calc_d1(spot_over_strike: float, vol: float, rate: float, time_to_expiry: float) -> float:
        return 1 / (vol * np.sqrt(time_to_expiry)) * (np.log(spot_over_strike) + (rate + vol**2 / 2) * time_to_expiry)

    @staticmethod
    def calc_d2(spot_over_strike: float, vol: float, rate: float, time_to_expiry: float) -> float:
        d1 = EuropeanAnalyticPricer.calc_d1(spot_over_strike, vol, rate, time_to_expiry)
        return d1 - vol * np.sqrt(time_to_expiry)

    def __init__(self, contract: EuropeanContract, model: MarketModel, params: Params) -> None:
        if not isinstance(contract, EuropeanContract):
            raise TypeError(f'Contract must be of type EuropeanContract but received {type(contract).__name__}')
        super().__init__(contract, model, params)

    def calc_fair_value(self) -> float:
        direction = self.contract.direction
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        spot = self.model.spot
        vol = self.model.vol
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)
        d1 = EuropeanAnalyticPricer.calc_d1(spot / strike, vol, rate, time_to_expiry)
        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)
        if self.contract.derivative_type == PutCallFwd.CALL:
            return direction * (spot * norm.cdf(d1) - strike * df * norm.cdf(d2))
        elif self.contract.derivative_type == PutCallFwd.PUT:
            return direction * (strike * df * norm.cdf(-d2) - spot * norm.cdf(-d1))
        else:
            self.contract.raise_incorrect_derivative_type_error()

    def calc_delta(self) -> float:
        greek = 0.0
        spot = self.model.spot
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        vol = self.model.vol
        rate = self.model.risk_free_rate
        d1 = EuropeanAnalyticPricer.calc_d1(spot / strike, vol, rate, time_to_expiry)
        if self.contract.derivative_type == PutCallFwd.CALL:
            greek = norm.cdf(d1)
        elif self.contract.derivative_type == PutCallFwd.PUT:
            greek = -norm.cdf(-d1)
        else:
            self.contract.raise_incorrect_derivative_type_error()
        return self.contract.direction * greek

    def calc_gamma(self) -> float:
        greek = 0.0
        spot = self.model.spot
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        vol = self.model.vol
        rate = self.model.risk_free_rate
        d1 = EuropeanAnalyticPricer.calc_d1(spot / strike, vol, rate, time_to_expiry)
        if self.contract.derivative_type in (PutCallFwd.CALL, PutCallFwd.PUT):
            greek = norm.pdf(d1) / (spot * vol * np.sqrt(time_to_expiry))
        else:
            self.contract.raise_incorrect_derivative_type_error()
        return self.contract.direction * greek

    def calc_vega(self) -> float:
        greek = 0.0
        spot = self.model.spot
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        vol = self.model.vol
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)
        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)
        if self.contract.derivative_type in (PutCallFwd.CALL, PutCallFwd.PUT):
            greek = strike * df * norm.pdf(d2) * np.sqrt(time_to_expiry)
        else:
            self.contract.raise_incorrect_derivative_type_error()
        return self.contract.direction * greek

    def calc_theta(self) -> float:
        greek = 0.0
        strike = self.contract.strike
        spot = self.model.spot
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        vol = self.model.vol
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)
        d1 = EuropeanAnalyticPricer.calc_d1(spot / strike, vol, rate, time_to_expiry)
        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)
        if self.contract.derivative_type == PutCallFwd.CALL:
            greek = -1.0 * ((spot * norm.pdf(d1) * vol) / (2 * np.sqrt(time_to_expiry))
                            + rate * strike * df * norm.cdf(d2))
        elif self.contract.derivative_type == PutCallFwd.PUT:
            greek = -1.0 * ((spot * norm.pdf(d1) * vol) / (2 * np.sqrt(time_to_expiry))
                            - rate * strike * df * norm.cdf(-d2))
        else:
            self.contract.raise_incorrect_derivative_type_error()
        return self.contract.direction * greek

    def calc_rho(self) -> float:
        greek = 0.0
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        spot = self.model.spot
        vol = self.model.vol
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)
        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)
        if self.contract.derivative_type == PutCallFwd.CALL:
            greek = strike * time_to_expiry * df * norm.cdf(d2)
        elif self.contract.derivative_type == PutCallFwd.PUT:
            greek = -strike * time_to_expiry * df * norm.cdf(-d2)
        else:
            self.contract.raise_incorrect_derivative_type_error()
        return self.contract.direction * greek


# =====================================================================================================
# Homework 3, Q1 (NEW CODE): NEW VERSION of EuropeanAnalyticPricer for EuropeanDigitalContract.
# Fair value (Lemma 4 of the report):  call  DF * N(d2),   put  DF * N(-d2) = DF - call   (DF = e^{-r tau})
# Greeks (Lemma 5): derivatives of DF * N(d2); the put follows from put = DF - call.
# d1, d2 are the static methods of EuropeanAnalyticPricer (reused, not copied).
# =====================================================================================================
# change # 1
#Delete.class EuropeanAnalyticPricer(Pricer):
# replace by
class EuropeanDigitalAnalyticPricer(Pricer):
# end change # 1
    # change # 2
    #Delete.def __init__(self, contract: EuropeanContract, model: MarketModel, params: Params) -> None:
    #Delete.    if not isinstance(contract, EuropeanContract):
    #Delete.        raise TypeError(f'Contract must be of type EuropeanContract but received {type(contract).__name__}')
    # replace by
    def __init__(self, contract: EuropeanDigitalContract, model: MarketModel, params: Params) -> None:
        if not isinstance(contract, EuropeanDigitalContract):
            raise TypeError(f'Contract must be of type EuropeanDigitalContract but received {type(contract).__name__}')
    # end change # 2
        super().__init__(contract, model, params)

    def calc_fair_value(self) -> float:
        direction = self.contract.direction
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        spot = self.model.spot
        vol = self.model.vol
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)
        # change # 3
        #Delete.d1 = EuropeanAnalyticPricer.calc_d1(spot / strike, vol, rate, time_to_expiry)
        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)
        #Delete.if self.contract.derivative_type == PutCallFwd.CALL:
        #Delete.    return direction * (spot * norm.cdf(d1) - strike * df * norm.cdf(d2))
        #Delete.elif self.contract.derivative_type == PutCallFwd.PUT:
        #Delete.    return direction * (strike * df * norm.cdf(-d2) - spot * norm.cdf(-d1))
        # replace by
        if self.contract.derivative_type == PutCallFwd.CALL:
            return direction * df * norm.cdf(d2)
        elif self.contract.derivative_type == PutCallFwd.PUT:
            return direction * df * norm.cdf(-d2)
        # end change # 3
        else:
            self.contract.raise_incorrect_derivative_type_error()

    def calc_delta(self) -> float:
        greek = 0.0
        spot = self.model.spot
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        vol = self.model.vol
        rate = self.model.risk_free_rate
        # change # 4
        #Delete.d1 = EuropeanAnalyticPricer.calc_d1(spot / strike, vol, rate, time_to_expiry)
        #Delete.if self.contract.derivative_type == PutCallFwd.CALL:
        #Delete.    greek = norm.cdf(d1)
        #Delete.elif self.contract.derivative_type == PutCallFwd.PUT:
        #Delete.    greek = -norm.cdf(-d1)
        # replace by
        df = self.model.calc_df(time_to_expiry)
        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)
        call_greek = df * norm.pdf(d2) / (spot * vol * np.sqrt(time_to_expiry))
        if self.contract.derivative_type == PutCallFwd.CALL:
            greek = call_greek
        elif self.contract.derivative_type == PutCallFwd.PUT:
            greek = -call_greek                                  # put = DF - call
        # end change # 4
        else:
            self.contract.raise_incorrect_derivative_type_error()
        return self.contract.direction * greek

    def calc_gamma(self) -> float:
        greek = 0.0
        spot = self.model.spot
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        vol = self.model.vol
        rate = self.model.risk_free_rate
        # change # 5
        #Delete.d1 = EuropeanAnalyticPricer.calc_d1(spot / strike, vol, rate, time_to_expiry)
        #Delete.if self.contract.derivative_type in (PutCallFwd.CALL, PutCallFwd.PUT):
        #Delete.    greek = norm.pdf(d1) / (spot * vol * np.sqrt(time_to_expiry))
        # replace by
        df = self.model.calc_df(time_to_expiry)
        d1 = EuropeanAnalyticPricer.calc_d1(spot / strike, vol, rate, time_to_expiry)
        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)
        call_greek = -df * norm.pdf(d2) * d1 / (spot**2 * vol**2 * time_to_expiry)
        if self.contract.derivative_type == PutCallFwd.CALL:
            greek = call_greek
        elif self.contract.derivative_type == PutCallFwd.PUT:
            greek = -call_greek
        # end change # 5
        else:
            self.contract.raise_incorrect_derivative_type_error()
        return self.contract.direction * greek

    def calc_vega(self) -> float:
        greek = 0.0
        spot = self.model.spot
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        vol = self.model.vol
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)
        # change # 6
        #Delete.d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)
        #Delete.if self.contract.derivative_type in (PutCallFwd.CALL, PutCallFwd.PUT):
        #Delete.    greek = strike * df * norm.pdf(d2) * np.sqrt(time_to_expiry)
        # replace by
        d1 = EuropeanAnalyticPricer.calc_d1(spot / strike, vol, rate, time_to_expiry)
        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)
        call_greek = -df * norm.pdf(d2) * d1 / vol
        if self.contract.derivative_type == PutCallFwd.CALL:
            greek = call_greek
        elif self.contract.derivative_type == PutCallFwd.PUT:
            greek = -call_greek
        # end change # 6
        else:
            self.contract.raise_incorrect_derivative_type_error()
        return self.contract.direction * greek

    def calc_theta(self) -> float:
        greek = 0.0
        strike = self.contract.strike
        spot = self.model.spot
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        vol = self.model.vol
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)
        d1 = EuropeanAnalyticPricer.calc_d1(spot / strike, vol, rate, time_to_expiry)
        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)
        # change # 7
        #Delete.if self.contract.derivative_type == PutCallFwd.CALL:
        #Delete.    greek = -1.0 * ((spot * norm.pdf(d1) * vol) / (2 * np.sqrt(time_to_expiry))
        #Delete.                    + rate * strike * df * norm.cdf(d2))
        #Delete.elif self.contract.derivative_type == PutCallFwd.PUT:
        #Delete.    greek = -1.0 * ((spot * norm.pdf(d1) * vol) / (2 * np.sqrt(time_to_expiry))
        #Delete.                    - rate * strike * df * norm.cdf(-d2))
        # replace by
        call_greek = (rate * df * norm.cdf(d2)
                      + df * norm.pdf(d2) * (d1 / (2 * time_to_expiry) - rate / (vol * np.sqrt(time_to_expiry))))
        if self.contract.derivative_type == PutCallFwd.CALL:
            greek = call_greek
        elif self.contract.derivative_type == PutCallFwd.PUT:
            greek = rate * df - call_greek                       # theta of DF = e^{-r(T-t)} is r * DF
        # end change # 7
        else:
            self.contract.raise_incorrect_derivative_type_error()
        return self.contract.direction * greek

    def calc_rho(self) -> float:
        greek = 0.0
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        spot = self.model.spot
        vol = self.model.vol
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)
        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)
        # change # 8
        #Delete.if self.contract.derivative_type == PutCallFwd.CALL:
        #Delete.    greek = strike * time_to_expiry * df * norm.cdf(d2)
        #Delete.elif self.contract.derivative_type == PutCallFwd.PUT:
        #Delete.    greek = -strike * time_to_expiry * df * norm.cdf(-d2)
        # replace by
        call_greek = -time_to_expiry * df * norm.cdf(d2) + df * norm.pdf(d2) * np.sqrt(time_to_expiry) / vol
        if self.contract.derivative_type == PutCallFwd.CALL:
            greek = call_greek
        elif self.contract.derivative_type == PutCallFwd.PUT:
            greek = -time_to_expiry * df - call_greek            # rho of DF = e^{-r tau} is -tau * DF
        # end change # 8
        else:
            self.contract.raise_incorrect_derivative_type_error()
        return self.contract.direction * greek
