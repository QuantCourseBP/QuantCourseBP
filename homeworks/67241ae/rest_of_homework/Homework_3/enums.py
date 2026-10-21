"""Reuse course enums when available; standalone fallback for this assignment."""
try:
    from src.enums import Stock, PutCallFwd, LongShort
except ModuleNotFoundError as exc:
    if exc.name not in ("src", "src.enums"):
        raise
    from enum import Enum

    class Stock(str, Enum):
        BLUECHIP_BANK = "BLUECHIP_BANK"
        TIPTOP_SOLUTIONS = "TIPTOP_SOLUTIONS"

    class PutCallFwd(str, Enum):
        CALL = "CALL"
        PUT = "PUT"
        FWD = "FWD"

    class LongShort(str, Enum):
        LONG = "LONG"
        SHORT = "SHORT"
