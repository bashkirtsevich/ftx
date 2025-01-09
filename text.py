# FIXME: Optimize; use string.digits


FT8_CHAR_TABLE_FULL = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-./?"
FT8_CHAR_TABLE_ALPHANUM_SPACE_SLASH = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/"
FT8_CHAR_TABLE_ALPHANUM_SPACE = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FT8_CHAR_TABLE_LETTERS_SPACE = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FT8_CHAR_TABLE_ALPHANUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FT8_CHAR_TABLE_NUMERIC = "0123456789"
FT8_CHAR_TABLE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# Convert integer index to ASCII character according to one of character tables
def charn(c: int, table: str) -> str:
    return table[c]


# Look up the index of an ASCII character in one of character tables
def nchar(c: str, table: str) -> int:
    return table.find(c)


def endswith_any(s: str, *tails: str) -> bool:
    return any(s.endswith(tail) for tail in tails)


def in_range(s: str, start: str, end: str) -> bool:
    return end >= s >= start
