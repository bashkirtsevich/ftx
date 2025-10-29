def byte(i: int) -> int:
    return i & 0xff


def dword(i: int) -> int:
    return i & 0xffffffff


def qword(i: int) -> int:
    return i & 0xffffffffffffffff
