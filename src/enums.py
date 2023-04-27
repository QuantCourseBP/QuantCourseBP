from enum import Enum


class Stock(str, Enum):
    # todo: replace EXAMPLE stocks with meaningful names
    EXAMPLE1: str = 'EXAMPLE1'
    EXAMPLE2: str = 'EXAMPLE2'


class PutCallFwd(str, Enum):
    PUT: str = 'PUT'
    CALL: str = 'CALL'
    FWD: str = 'FWD'
