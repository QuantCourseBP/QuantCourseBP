from __future__ import annotations
from src.contract import *
from src.enums import *
from src.model import *
from src.numerical_method import *
from src.pricer import *


class CalcEngine:
    __SUPPORTED_MODELS: list[str] = list(MarketModel.get_models().keys())
    __SUPPORTED_NUMERICAL_METHODS: list[str] = list(NumericalMethod.get_numerical_methods().keys())
    __GENERIC_NUMERICAL_METHODS: list[str] = [
        TreeMethod.__name__,
        PDEMethod.__name__,
        MCMethod.__name__
    ]

    def __init__(self, contracts: list[Contract], model_name: str, method_name: str, params: Params) -> None:
        self.__contracts: list[Contract] = contracts
        self.__model_name: str = model_name
        self.__numerical_method_name: str = method_name
        self.__params: Params = params
        self.__validate()
        self.__pricers: dict[Contract, Pricer] = self.__create_pricers()

    def __validate(self) -> None:
        if type(self.__contracts) != list or not all([isinstance(obj, Contract) for obj in self.__contracts]):
            raise ValueError(f'Attribute `contracts` must be a list of Contract objects')

        if self.__model_name not in self.__SUPPORTED_MODELS:
            raise ValueError(f'Unsupported market model: {self.__model_name}, '
                             f'choose from {self.__SUPPORTED_MODELS}')

        if self.__numerical_method_name not in self.__SUPPORTED_NUMERICAL_METHODS:
            raise ValueError(f'Unsupported numerical method: {self.__numerical_method_name}, '
                             f'choose from {self.__SUPPORTED_NUMERICAL_METHODS}')

        if not isinstance(self.__params, Params):
            raise ValueError(f'Attribute `params` must be a Params objects')

    def __create_pricers(self) -> dict[Contract, Pricer]:
        pricers = dict()
        for contract in self.__contracts:
            model = globals()[self.__model_name](contract.get_underlying())
            if self.__numerical_method_name in self.__GENERIC_NUMERICAL_METHODS:
                contract = contract.convert_to_generic()
            contract_name = contract.__class__.__name__.removesuffix('Contract')
            method_name = self.__numerical_method_name.removesuffix('Method')
            pricer_name = f'{contract_name}{method_name}Pricer'
            if self.__numerical_method_name == AnalyticMethod.__name__:
                pricer = globals()[pricer_name](contract, model)
            else:
                numerical_method = globals()[self.__numerical_method_name](model, self.__params)
                pricer = globals()[pricer_name](contract, model, numerical_method)
            pricers[contract] = pricer
        return pricers

    def get_pricers(self) -> dict[Contract, Pricer]:
        return self.__pricers

    # todo: to be implemented
    def calculate(self) -> ModelResult:
        pass


# todo: to be implemented
class ModelResult:
    pass
