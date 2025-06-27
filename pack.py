import re
import typing
from functools import reduce

from consts import CALLSIGN_HASHTABLE_SIZE, FTX_RESPONSE_EXTRAS_CODE, FTX_MAX_GRID_4, FTX_RESPONSE_EXTRAS_STR
from consts import FTX_TOKEN_CODE
from consts import FTX_TOKEN_STR
from consts import FTX_CALLSIGN_HASH_10_BITS
from consts import FTX_CALLSIGN_HASH_12_BITS
from consts import FTX_CALLSIGN_HASH_22_BITS
from exceptions import FTXInvalidCallsign, FTXErrorGrid, FTXInvalidRST
from exceptions import FTXPack28Error
from text import FTX_CHAR_TABLE_ALPHANUM
from text import FTX_GRID_CHAR_MAP
from text import FTX_BASECALL_CHAR_MAP
from text import FTX_CHAR_TABLE_ALPHANUM_SPACE
from text import FTX_CHAR_TABLE_ALPHANUM_SPACE_SLASH
from text import FTX_CHAR_TABLE_LETTERS
from text import FTX_CHAR_TABLE_LETTERS_SPACE
from text import FTX_CHAR_TABLE_NUMERIC
from text import charn
from text import endswith_any
from text import in_range
from text import nchar
from tools import dword, qword

NTOKENS = 2063592
MAX22 = 4194304


def save_callsign(callsign: str) -> typing.Optional[typing.Tuple[int, int, int]]:
    n58 = 0
    i = 0
    for c in callsign:
        j = nchar(c, FTX_CHAR_TABLE_ALPHANUM_SPACE_SLASH)
        if j < 0:
            return None  # hash error (wrong character set)
        n58 = 38 * n58 + j
        i += 1

    # pretend to have trailing whitespace (with j=0, index of ' ')
    while i < 11:
        n58 = 38 * n58
        i += 1

    n22 = ((47055833459 * n58) >> (64 - 22)) & 0x3FFFFF
    n12 = n22 >> 10
    n10 = n22 >> 12
    # LOG(LOG_DEBUG, "save_callsign('%s') = [n22=%d, n12=%d, n10=%d]\n", callsign, n22, n12, n10)

    # if (hash_if != NULL)
    #     hash_if->save_hash(callsign, n22)
    # print(f"save_callsign({callsign}) =", dword(n22), dword(n12), dword(n10))
    return dword(n22), dword(n12), dword(n10)


def lookup_hash(hash_type: int, hash: int) -> typing.Optional[str]:
    hash_shift = 12 if hash_type == FTX_CALLSIGN_HASH_10_BITS else 10 if hash_type == FTX_CALLSIGN_HASH_12_BITS else 0
    hash10 = (hash >> (12 - hash_shift)) & 0x3FF
    idx_hash = (hash10 * 23) % CALLSIGN_HASHTABLE_SIZE
    # while callsign_hashtable[idx_hash].callsign[0]:
    #     if ((callsign_hashtable[idx_hash].hash & 0x3FFFFF) >> hash_shift) == hash:
    #         return callsign_hashtable[idx_hash].callsign
    #     #  Move on to check the next entry in hash table
    #     idx_hash = (idx_hash + 1) % CALLSIGN_HASHTABLE_SIZE
    return None


def lookup_callsign(hash_type: int, hash: int) -> str:
    c11 = lookup_hash(hash_type, hash)
    # LOG(LOG_DEBUG, "lookup_callsign(n%s=%d) = '%s'\n", (hash_type == FTX_CALLSIGN_HASH_22_BITS ? "22" : (hash_type == FTX_CALLSIGN_HASH_12_BITS ? "12" : "10")), hash, callsign);
    return f"<{c11 if c11 else '...'}>"


def pack_basecall(callsign: str) -> int:
    if (length := len(callsign)) > 2:
        # Work-around for Swaziland prefix: 3DA0XYZ -> 3D0XYZ
        if callsign.startswith("3DA0") and 4 < length <= 7:
            cs_6 = f"3D0{callsign[4:]}"
        # Work-around for Guinea prefixes: 3XA0XYZ -> QA0XYZ
        elif callsign.startswith("3X") and callsign[2].isalpha() and length <= 7:
            cs_6 = f"Q{callsign[2:]}"
        elif callsign[2].isdigit() and length <= 6:
            cs_6 = callsign
        # Check the position of callsign digit and make a right-aligned copy into cs_6
        elif callsign[1].isdigit() and length <= 5:
            # A0XYZ -> " A0XYZ"
            cs_6 = f" {callsign}"
        else:
            cs_6 = " " * 6

        cs_6 = cs_6 + " " * (6 - len(cs_6))  # Normalize to 6 letters

        # Check for standard callsign
        n_chars = list(map(nchar, cs_6, FTX_BASECALL_CHAR_MAP))

        if all(nc >= 0 for nc in n_chars):
            # This is a standard callsign
            # LOG(LOG_DEBUG, "Encoding basecall [%.6s]\n", cs_6);
            n = reduce(lambda a, it: a * len(it[0]) + it[1], zip(FTX_BASECALL_CHAR_MAP, n_chars), 0)
            return n  # Standard callsign
    return -1


def pack_grid(grid4: str) -> int:
    n_chars = list(map(nchar, grid4, FTX_GRID_CHAR_MAP))
    n = reduce(lambda a, it: a * len(it[0]) + it[1], zip(FTX_GRID_CHAR_MAP, n_chars), 0)
    return n  # Standard callsign


def pack_extra(extra: str) -> int:
    if id_resp := FTX_RESPONSE_EXTRAS_CODE.get(extra):
        return id_resp

    # Check for standard 4 letter grid
    if re.match(r"^(([A-R]{2})([0-9]{2}))$", extra):
        return pack_grid(extra)

    # Parse report: +dd / -dd / R+dd / R-dd
    if not (report := re.match(r"^(R){0,1}([\+\-]{0,1}[0-9]+)$", extra)):
        raise FTXInvalidRST

    r_sign, r_val = report.groups()
    i_report = int(r_val) + 35
    return (FTX_MAX_GRID_4 + i_report) | (0x8000 if r_sign is not None else 0)


def pack28(callsign: str) -> typing.Tuple[int, int]:
    shift = 0
    # Check for special tokens first
    if token := FTX_TOKEN_CODE.get(callsign):
        return token, shift

    length = len(callsign)
    if callsign.startswith("CQ_") and length < 8:
        rest = callsign[3:]
        rest_len = len(rest)

        if rest_len == 3 and rest.isdigit():
            return int(rest) + 3, shift

        if 1 <= rest_len <= 4:
            nlet = 0
            correct = True
            for c in rest:
                if (n := nchar(c, FTX_CHAR_TABLE_LETTERS_SPACE)) == -1:
                    correct = False
                    break
                nlet = nlet * 27 + n

            if correct:
                return nlet + 1003, shift

    # Detect /R and /P suffix for basecall check
    length_base = length
    if endswith_any(callsign, "/P", "/R"):
        # LOG(LOG_DEBUG, "Suffix /P or /R detected\n");
        shift = 1
        length_base = length - 2

    if (n28 := pack_basecall(callsign[:length_base])) >= 0:
        # Callsign can be encoded as a standard basecall with optional /P or /R suffix
        if save_callsign(callsign) is None:
            raise FTXInvalidCallsign  # Error (some problem with callsign contents)
        return dword(NTOKENS + MAX22 + n28), shift  # Standard callsign

    if 3 < length <= 11:
        # Treat this as a nonstandard callsign: compute its 22-bit hash
        # LOG(LOG_DEBUG, "Encoding as non-standard callsign\n");

        if (x := save_callsign(callsign)) is None:
            raise FTXInvalidCallsign  # Error (some problem with callsign contents)
        shift = 0
        n22, _, _ = x
        return dword(NTOKENS + n22), shift  # 22-bit hashed callsign

    raise FTXPack28Error


def pack58(callsign: str) -> typing.Optional[int]:
    # Decode one of the calls from 58 bit encoded string
    # const char* src = callsign;
    if callsign.startswith("<"):
        callsign = callsign[1:]

    result = 0
    c11 = ""
    for i, c in enumerate(callsign):
        if c == "<" or i >= 11:
            break

        c11 += c
        j = nchar(c, FTX_CHAR_TABLE_ALPHANUM_SPACE_SLASH)
        if j < 0:
            return None
        result = qword((result * 38) + j)

    if save_callsign(c11) is None:
        return None

    # LOG(LOG_DEBUG, "pack58('%s')=%016llx\n", callsign, *n58);
    # print(f"pack58({callsign}) =", hex(result))
    return result


def unpack28(n28: int, ip: int, i3: int) -> typing.Optional[str]:
    # LOG(LOG_DEBUG, "unpack28() n28=%d i3=%d\n", n28, i3);
    # Check for special tokens DE, QRZ, CQ, CQ_nnn, CQ_aaaa
    if n28 < NTOKENS:
        if n28 <= 2:
            return FTX_TOKEN_STR.get(n28)

        if n28 <= 1002:
            # CQ nnn with 3 digits
            return f"CQ_{n28 - 3:03}"

        if n28 <= 532443:
            # CQ ABCD with 4 alphanumeric symbols
            n = n28 - 1003
            aaaa = ""
            for i in range(4):
                aaaa = charn(n % 27, FTX_CHAR_TABLE_LETTERS_SPACE) + aaaa
                n //= 27
            return f"CQ_{aaaa.strip()}"

        # unspecified
        return None

    n28 -= NTOKENS
    if n28 < MAX22:
        # This is a 22-bit hash of a result
        return lookup_callsign(FTX_CALLSIGN_HASH_22_BITS, n28)

    # Standard callsign
    n = n28 - MAX22
    callsign = charn(n % 27, FTX_CHAR_TABLE_LETTERS_SPACE)
    n //= 27
    callsign = charn(n % 27, FTX_CHAR_TABLE_LETTERS_SPACE) + callsign
    n //= 27
    callsign = charn(n % 27, FTX_CHAR_TABLE_LETTERS_SPACE) + callsign
    n //= 27
    callsign = charn(n % 10, FTX_CHAR_TABLE_NUMERIC) + callsign
    n //= 10
    callsign = charn(n % 36, FTX_CHAR_TABLE_ALPHANUM) + callsign
    n //= 36
    callsign = charn(n % 37, FTX_CHAR_TABLE_ALPHANUM_SPACE) + callsign

    callsign = callsign.strip()

    # Copy callsign to 6 character buffer
    if callsign.startswith("3D0") and callsign[3] != " ":
        # Work-around for Swaziland prefix: 3D0XYZ -> 3DA0XYZ
        result = f"3DA0{callsign[3:]}"
    elif callsign[0] == "Q" and callsign[1].isalpha():
        # Work-around for Guinea prefixes: QA0XYZ -> 3XA0XYZ
        result = f"3X{callsign[1:]}"
    else:
        # Skip trailing and leading whitespace in case of a short callsign
        result = callsign

    length = len(result)
    if length < 3:
        return None  # Callsign too short

    # Check if we should append /R or /P suffix
    if ip:
        # FIXME: Optimize
        if i3 == 1:
            result = f"{result}/R"
        elif i3 == 2:
            result = f"{result}/P"
        else:
            raise ValueError

    # Save the result to hash table
    save_callsign(result)

    return result


def unpackgrid(igrid4: int, ir: int) -> typing.Optional[str]:
    if igrid4 <= FTX_MAX_GRID_4:
        # Extract 4 symbol grid locator
        n = igrid4

        # FIXME: Optimize
        dst = charn(n % 10, FTX_CHAR_TABLE_NUMERIC)  # 0..9
        n //= 10
        dst = charn(n % 10, FTX_CHAR_TABLE_NUMERIC) + dst  # 0..9
        n //= 10
        dst = charn(n % 18, FTX_CHAR_TABLE_LETTERS) + dst  # A..R
        n //= 18
        dst = charn(n % 18, FTX_CHAR_TABLE_LETTERS) + dst  # A..R

        # In case of ir=1 add an "R " before grid
        return f"{'R ' if ir else ''}{dst}"
    else:
        # Extract report
        if irpt := FTX_RESPONSE_EXTRAS_STR.get(igrid4):
            return irpt
        # irpt = igrid4 - FTX_MAX_GRID_4
        #
        # # Check special cases first (irpt > 0 always)
        # if irpt == 1:
        #     return ""
        # elif irpt == 2:
        #     return "RRR"
        # elif irpt == 3:
        #     return "RR73"
        # elif irpt == 4:
        #     return "73"
        else:
            # Extract signal report as a two digit number with a + or - sign
            return f"{'R' if ir else ''}{int(irpt - 35):+03}"


def unpack58(n58: int) -> str:
    #  Decode one of the calls from 58 bit encoded string
    c11 = ""
    for i in range(10):
        c11 = charn(n58 % 38, FTX_CHAR_TABLE_ALPHANUM_SPACE_SLASH) + c11
        n58 //= 38

    # The decoded string will be right-aligned, so trim all whitespace (also from back just in case)
    callsign = c11.strip()

    # LOG(LOG_DEBUG, "unpack58(%016llx)=%s\n", n58_backup, callsign);

    # Save the decoded call in a hash table for later
    # if len(callsign) >= 3:
    #     return save_callsign(callsign)

    return callsign
