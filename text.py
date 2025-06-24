import string

FTX_CHAR_TABLE_NUMERIC = string.digits
FTX_CHAR_TABLE_LETTERS = string.ascii_uppercase
FTX_CHAR_TABLE_GRID_LETTERS = FTX_CHAR_TABLE_LETTERS[:18]
FTX_CHAR_TABLE_ALPHANUM = f"{FTX_CHAR_TABLE_NUMERIC}{FTX_CHAR_TABLE_LETTERS}"
FTX_CHAR_TABLE_LETTERS_SPACE = f" {FTX_CHAR_TABLE_LETTERS}"
FTX_CHAR_TABLE_ALPHANUM_SPACE = f" {FTX_CHAR_TABLE_ALPHANUM}"
FTX_CHAR_TABLE_ALPHANUM_SPACE_SLASH = f"{FTX_CHAR_TABLE_ALPHANUM_SPACE}/"
FTX_CHAR_TABLE_FULL = f"{FTX_CHAR_TABLE_ALPHANUM_SPACE}+-./?"

FTX_BASECALL_CHAR_MAP = [
    FTX_CHAR_TABLE_ALPHANUM_SPACE,
    FTX_CHAR_TABLE_ALPHANUM,
    FTX_CHAR_TABLE_NUMERIC,
    FTX_CHAR_TABLE_LETTERS_SPACE,
    FTX_CHAR_TABLE_LETTERS_SPACE,
    FTX_CHAR_TABLE_LETTERS_SPACE
]

FTX_GRID_CHAR_MAP = [
    FTX_CHAR_TABLE_GRID_LETTERS,
    FTX_CHAR_TABLE_GRID_LETTERS,
    FTX_CHAR_TABLE_NUMERIC,
    FTX_CHAR_TABLE_NUMERIC
]


def charn(c: int, table: str) -> str:
    # Convert integer index to ASCII character according to one of character tables
    return table[c]


def nchar(c: str, table: str) -> int:
    # Look up the index of an ASCII character in one of character tables
    return table.find(c)


def endswith_any(s: str, *tails: str) -> bool:
    return any(s.endswith(tail) for tail in tails)


def in_range(s: str, start: str, end: str) -> bool:
    return end >= s >= start
