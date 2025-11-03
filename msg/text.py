import string
import typing
from functools import reduce, partial

from msg.exceptions import CharTableMissmatch

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

FTX_BASECALL_SUFFIX_FMT = {
    1: "{cs}/R",
    2: "{cs}/P",
}


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


def ct_validate(ct: str, val: str, raise_exception: bool = False) -> bool:
    x = all(c in ct for c in val)

    if not x and raise_exception:
        raise CharTableMissmatch("Character table missmatch")

    return x


def ct_encode(ct: str, val: str) -> int:
    return reduce(lambda a, j: len(ct) * a + j, map(partial(nchar, table=ct), val))


def ct_decode(ct: str, val: int, l: int) -> str:
    s = ""
    ct_l = len(ct)
    for i in range(l):
        s = charn(val % ct_l, ct) + s
        val //= ct_l

    return s


def ct_map_encode(ct_map: typing.List[str], val: str) -> int:
    n_chars = map(nchar, val, ct_map)
    n_ct_len = map(len, ct_map)

    return reduce(lambda a, it: a * it[0] + it[1], zip(n_ct_len, n_chars), 0)


def ct_map_decode(ct_map: typing.List[str], val: int) -> str:
    s = ""
    for ct_len, ct in map(lambda ct: (len(ct), ct), reversed(ct_map)):
        s = charn(val % ct_len, ct) + s
        val //= ct_len

    return s
