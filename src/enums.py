from enum import Enum


class Stock(str, Enum):
    # todo: replace EXAMPLE stocks with meaningful names
    EXAMPLE1: str = 'EXAMPLE1'
    EXAMPLE2: str = 'EXAMPLE2'


class PutCallFwd(str, Enum):
    PUT: str = 'PUT'
    CALL: str = 'CALL'
    FWD: str = 'FWD'


class Measure(str, Enum):
    FAIR_VALUE: str = 'FAIR_VALUE'
    DELTA: str = 'DELTA'
    GAMMA: str = 'GAMMA'
    VEGA: str = 'VEGA'
    THETA: str = 'THETA'
    RHO: str = 'RHO'


class GreekMethod(str, Enum):
    ANALYTIC: str = 'ANALYTIC'
    BUMP: str = 'BUMP'
