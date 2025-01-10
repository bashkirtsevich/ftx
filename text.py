import string

FT8_CHAR_TABLE_NUMERIC = string.digits
FT8_CHAR_TABLE_LETTERS = string.ascii_uppercase
FT8_CHAR_TABLE_ALPHANUM = f"{FT8_CHAR_TABLE_NUMERIC}{FT8_CHAR_TABLE_LETTERS}"
FT8_CHAR_TABLE_LETTERS_SPACE = f" {FT8_CHAR_TABLE_LETTERS}"
FT8_CHAR_TABLE_ALPHANUM_SPACE = f" {FT8_CHAR_TABLE_ALPHANUM}"
FT8_CHAR_TABLE_ALPHANUM_SPACE_SLASH = f"{FT8_CHAR_TABLE_ALPHANUM_SPACE}/"
FT8_CHAR_TABLE_FULL = f"{FT8_CHAR_TABLE_ALPHANUM_SPACE}+-./?"


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
