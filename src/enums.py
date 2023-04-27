from enum import Enum


class Stock(str, Enum):
    # todo: replace EXAMPLE stocks with meaningful names
    EXAMPLE1: str = 'EXAMPLE1'
    EXAMPLE2: str = 'EXAMPLE2'


class PutCallFwd(str, Enum):
    PUT: str = 'PUT'
    CALL: str = 'CALL'
    FWD: str = 'FWD'


class LongShort(str, Enum):
    LONG: str = 'LONG'
    SHORT: str = 'SHORT'


class UpDown(str, Enum):     # for Barrier Contract
    UP: str = 'UP'
    DOWN: str = 'DOWN'


class InOut(str, Enum):     # for Barrier Contract
    IN: str = 'IN'
    OUT: str = 'OUT'




