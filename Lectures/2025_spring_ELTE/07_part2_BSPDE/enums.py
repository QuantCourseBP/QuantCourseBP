from enum import Enum

class BSPDEMethod(str, Enum):
    EXPLICIT: str = 'EXPLICIT'
    IMPLICIT: str = 'IMPLICIT'
    CRANKNICOLSON: str = 'CRANKNICOLSON'


