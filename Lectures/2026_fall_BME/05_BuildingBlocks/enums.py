from enum import Enum


class Stock(str, Enum):
    TEST_COMPANY: str = 'TEST_COMPANY'
    BLUECHIP_BANK: str = 'BLUECHIP_BANK'
    TIPTOP_SOLUTIONS: str = 'TIPTOP_SOLUTIONS'
    MONEY_MALL: str = 'MONEY_MALL'


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


class LongShort(str, Enum):
    LONG: str = 'LONG'
    SHORT: str = 'SHORT'


class UpDown(str, Enum):  # for Barrier Contract
    UP: str = 'UP'
    DOWN: str = 'DOWN'


class InOut(str, Enum):  # for Barrier Contract
    IN: str = 'IN'
    OUT: str = 'OUT'


class BSPDEMethod(str, Enum):
    EXPLICIT: str = 'EXPLICIT'
    IMPLICIT: str = 'IMPLICIT'
    CRANK_NICOLSON: str = 'CRANK_NICOLSON'
