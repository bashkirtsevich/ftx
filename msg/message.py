import re
import typing
from abc import ABCMeta, abstractmethod
from contextlib import suppress
from functools import reduce, partial

from consts.msg import MSG_CALLSIGN_HASH_12_BITS, MSG_MESSAGE_TYPE_FREE_TEXT, MSG_MESSAGE_TYPE_DXPEDITION, \
    MSG_MESSAGE_TYPE_EU_VHF, MSG_MESSAGE_TYPE_ARRL_FD, MSG_MESSAGE_TYPE_TELEMETRY, MSG_MESSAGE_TYPE_STANDARD, \
    MSG_MESSAGE_TYPE_ARRL_RTTY, MSG_MESSAGE_TYPE_NONSTD_CALL, MSG_MESSAGE_TYPE_WWROF, MSG_MESSAGE_TYPE_UNKNOWN, \
    MSG_MESSAGE_FREE_TEXT_LEN, MSG_MESSAGE_TELEMETRY_LEN
from consts.ftx import FTX_EXTRAS_CODE, FTX_MAX_GRID_4
from consts.ftx import FTX_EXTRAS_STR
from .exceptions import MSGErrorCallSignTo, MSGErrorTooLong, MSGErrorInvalidChar, MSGException
from .exceptions import MSGErrorCallSignDe
from .exceptions import MSGErrorGrid
from .exceptions import MSGErrorMsgType
from .exceptions import MSGErrorSuffix
from .pack import pack_callsign, save_callsign, pack_extra, pack58, unpack_callsign, unpack_extra, lookup_callsign, \
    unpack58, \
    pack_basecall, pack_grid
from .text import FTX_CHAR_TABLE_FULL, charn, nchar, endswith_any, FTX_CHAR_TABLE_ALPHANUM_SPACE_SLASH, \
    FTX_GRID_CHAR_MAP
from .tools import byte, dword


class Item(metaclass=ABCMeta):
    __slots__ = ("val_str", "val_int")

    def __init__(self, val: typing.Union[str, int]):
        if isinstance(val, str):
            self.val_str = val.strip()
            self.val_int = self.to_int()
        elif isinstance(val, int):
            self.val_int = val
            self.val_str = self.to_str()
        else:
            raise TypeError(f"Unsupported data type {type(val)}")

    @abstractmethod
    def to_int(self) -> int:
        ...

    @abstractmethod
    def to_str(self) -> str:
        ...

    @property
    def as_str(self):
        return self.val_str

    @property
    def as_int(self):
        return self.val_int

    def __str__(self):
        return self.val_str

    def __repr__(self):
        return str(self)


class Callsign(Item):
    def to_int(self) -> int:
        if any(c not in FTX_CHAR_TABLE_ALPHANUM_SPACE_SLASH for c in self.val_str):
            raise ValueError("Invalid characters")

        return pack_callsign(self.val_str)[0]

    def to_str(self) -> str:
        return unpack_callsign(self.val_int, False, 0)

    def hash_22(self):
        ct = FTX_CHAR_TABLE_ALPHANUM_SPACE_SLASH

        acc = reduce(lambda a, j: len(ct) * a + j, map(partial(nchar, table=ct), self.val_str))

        # pretend to have trailing whitespace (with j=0, index of ' ')
        if (val_len := len(self.val_str)) < 11:
            acc *= len(ct) ** (11 - val_len)

        hash = ((47055833459 * acc) >> (64 - 22)) & 0x3fffff
        return hash

    def hash_12(self):
        hash = self.hash_22()
        return hash >> 10

    def hash_10(self):
        hash = self.hash_22()
        return hash >> 12

    def __hash__(self):
        return self.hash_22()


class Grid(Item):
    def to_int(self) -> int:
        if len(self.val_str) != 4:
            raise ValueError("Invalid grid descriptor length")

        if any(True for c, ct in zip(self.val_str, FTX_GRID_CHAR_MAP) if c not in ct):
            raise ValueError("Invalid grid descriptor character")

        n_chars = map(nchar, self.val_str, FTX_GRID_CHAR_MAP)
        n_ct_len = map(len, FTX_GRID_CHAR_MAP)
        return reduce(lambda a, it: a * it[0] + it[1], zip(n_ct_len, n_chars), 0)

    def to_str(self) -> str:
        if self.val_int > FTX_MAX_GRID_4:
            raise ValueError("Invalid grid descriptor value")

        n = self.val_int
        val = ""
        for ct_len, ct in map(lambda ct: (len(ct), ct), reversed(FTX_GRID_CHAR_MAP)):
            val = charn(n % ct_len, ct) + val
            n //= ct_len

        return val


class Report(Item):
    def to_int(self) -> int:
        if not (report := re.match(r"^(R){0,1}([\+\-]{0,1})([0-9]+)$", self.val_str)):
            raise ValueError("Invalid report value")

        _, sign, val = report.groups()

        report = int(val) + 35
        return (FTX_MAX_GRID_4 + report) | (0x8000 if sign == "-" else 0)

    def to_str(self) -> str:
        if self.val_int <= FTX_MAX_GRID_4:
            raise ValueError("Invalid report representation")

        val = int(self.val_int - FTX_MAX_GRID_4 - 35)
        if val & 0x8000:
            val = -(val & 0x7fff)

        return f"R{val:+03}"


def message_encode(call_to: str, call_de: str, extra: str = "") -> typing.ByteString:
    if len(call_to) > 11:
        raise MSGErrorCallSignTo

    if len(call_de) > 11:
        raise MSGErrorCallSignDe

    if len(extra) > 19:
        raise MSGErrorGrid

    with suppress(MSGException):
        return message_encode_std(call_to, call_de, extra)

    with suppress(MSGException):
        return message_encode_nonstd(call_to, call_de, extra)

    with suppress(MSGException):
        return message_encode_free(call_to)


def message_decode(payload: typing.ByteString) -> typing.Tuple[str, typing.Optional[str], typing.Optional[str]]:
    msg_type = message_get_type(payload)
    if msg_type == MSG_MESSAGE_TYPE_STANDARD:
        field1, field2, field3 = message_decode_std(payload)
    elif msg_type == MSG_MESSAGE_TYPE_NONSTD_CALL:
        field1, field2, field3 = message_decode_nonstd(payload)
    elif msg_type == MSG_MESSAGE_TYPE_FREE_TEXT:
        field1 = message_decode_free(payload)
        field2 = None
        field3 = None
    elif msg_type == MSG_MESSAGE_TYPE_TELEMETRY:
        field1 = message_decode_telemetry_hex(payload)
        field2 = None
        field3 = None
    else:
        # not handled yet
        raise MSGErrorMsgType

    return field1, field2, field3


def message_get_type(payload: typing.ByteString) -> int:
    # Extract i3 (bits 74..76)
    # FIXME: Optimize, use dict instead
    i3 = (payload[9] >> 3) & 0x07
    if i3 == 0:
        # Extract n3 (bits 71..73)
        n3 = ((payload[8] << 2) & 0x04) | ((payload[9] >> 6) & 0x03)
        if n3 == 0:
            return MSG_MESSAGE_TYPE_FREE_TEXT
        elif n3 == 1:
            return MSG_MESSAGE_TYPE_DXPEDITION
        elif n3 == 2:
            return MSG_MESSAGE_TYPE_EU_VHF
        elif n3 < 5:
            return MSG_MESSAGE_TYPE_ARRL_FD
        elif n3 == 5:
            return MSG_MESSAGE_TYPE_TELEMETRY
        else:
            return MSG_MESSAGE_TYPE_UNKNOWN
    elif i3 < 3:
        return MSG_MESSAGE_TYPE_STANDARD
    elif i3 == 3:
        return MSG_MESSAGE_TYPE_ARRL_RTTY
    elif i3 == 4:
        return MSG_MESSAGE_TYPE_NONSTD_CALL
    elif i3 == 5:
        return MSG_MESSAGE_TYPE_WWROF
    else:
        return MSG_MESSAGE_TYPE_UNKNOWN


def message_encode_std(call_to: str, call_de: str, extra: str) -> typing.ByteString:
    b28_to, sh_to = pack_callsign(call_to)
    if b28_to < 0:
        raise MSGErrorCallSignTo

    b28_de, sh_de = pack_callsign(call_de)
    if b28_de < 0:
        raise MSGErrorCallSignDe

    suffix = 1  # No suffix or /R
    if any(cs.endswith("/P") for cs in (call_to, call_de)):
        suffix = 2  # Suffix /P for EU VHF contest
        if any(cs.endswith("/R") for cs in (call_to, call_de)):
            raise MSGErrorSuffix

    if call_to == "CQ" and "/" in call_de and not endswith_any(call_de, "/P", "/R"):
        raise MSGErrorCallSignDe  # nonstandard call: need a type 4 message

    b16_extra = pack_extra(extra)

    # Shift in sh_a and sh_b bits into n28a and n28b
    b29_to = dword(b28_to << 1 | sh_to)
    b29_de = dword(b28_de << 1 | sh_de)

    # TODO: check for suffixes
    if endswith_any(call_to, "/P", "/R"):
        b29_to |= 1  # sh_a = 11
        if call_to.endswith("/P"):
            suffix = 2

    # Pack into (28 + 1) + (28 + 1) + (1 + 15) + 3 bits
    bytes = [
        byte(b29_to >> 21),
        byte(b29_to >> 13),
        byte(b29_to >> 5),
        byte(b29_to << 3) | byte(b29_de >> 26),
        byte(b29_de >> 18),
        byte(b29_de >> 10),
        byte(b29_de >> 2),
        byte(b29_de << 6) | byte(b16_extra >> 10),
        byte(b16_extra >> 2),
        byte(b16_extra << 6) | byte(suffix << 3)
    ]
    return bytearray(b for b in bytes)


def message_decode_std(payload: typing.ByteString) -> typing.Tuple[str, str, str]:
    # Extract packed fields
    b29_to = payload[0] << 21
    b29_to |= payload[1] << 13
    b29_to |= payload[2] << 5
    b29_to |= payload[3] >> 3

    b29_de = (payload[3] & 0x07) << 26
    b29_de |= payload[4] << 18
    b29_de |= payload[5] << 10
    b29_de |= payload[6] << 2
    b29_de |= payload[7] >> 6

    r_flag = (payload[7] & 0x20) >> 5

    b16_extra = (payload[7] & 0x1F) << 10
    b16_extra |= payload[8] << 2
    b16_extra |= payload[9] >> 6

    # Extract cs_flags (bits 74..76)
    cs_flags = (payload[9] >> 3) & 0x07

    if (call_to := unpack_callsign(b29_to >> 1, bool(b29_to & 1), cs_flags)) is None:
        raise MSGErrorCallSignTo

    if (call_de := unpack_callsign(b29_de >> 1, bool(b29_de & 1), cs_flags)) is None:
        raise MSGErrorCallSignDe

    if (extra := unpack_extra(b16_extra, bool(r_flag & 1))) is None:
        raise MSGErrorGrid

    return call_to, call_de, extra


def message_decode_nonstd(payload: typing.ByteString) -> typing.Tuple[str, str, str]:
    # non-standard messages, code originally by KD8CEC
    n12 = payload[0] << 4  # 11 ~ 4 : 8
    n12 |= payload[1] >> 4  # 3 ~ 0  : 12

    n58 = (payload[1] & 0x0F) << 54  # 57 ~ 54 : 4
    n58 |= payload[2] << 46  # 53 ~ 46 : 12
    n58 |= payload[3] << 38  # 45 ~ 38 : 12
    n58 |= payload[4] << 30  # 37 ~ 30 : 12
    n58 |= payload[5] << 22  # 29 ~ 22 : 12
    n58 |= payload[6] << 14  # 21 ~ 14 : 12
    n58 |= payload[7] << 6  # 13 ~ 6  : 12
    n58 |= payload[8] >> 2  # 5 ~ 0   : 765432 10

    iflip = (payload[8] >> 1) & 0x01  # 76543210
    nrpt = (payload[8] & 0x01) << 1
    nrpt |= payload[9] >> 7  # 76543210
    icq = (payload[9] >> 6) & 0x01

    # Extract i3 (bits 74..76)
    # i3 = (payload[9] >> 3) & 0x07  # UNUSED

    # Decode one of the calls from 58 bit encoded string
    call_decoded = unpack58(n58)

    # Decode the other call from hash lookup table
    call_3 = lookup_callsign(MSG_CALLSIGN_HASH_12_BITS, n12)

    # Possibly flip them around
    call_1 = call_decoded if iflip else call_3
    call_2 = call_3 if iflip else call_decoded

    if not icq:
        call_to = call_1

        extra = FTX_EXTRAS_STR.get(nrpt, "")
    else:
        call_to = "CQ"
        extra = ""

    call_de = call_2

    return call_to, call_de, extra


def message_encode_nonstd(call_to: str, call_de: str, extra: str) -> typing.ByteString:
    i3 = 4

    icq = call_to == "CQ"
    len_call_to = len(call_to)
    len_call_de = len(call_de)

    if not icq and len_call_to < 3:
        raise MSGErrorCallSignTo

    if len_call_de < 3:
        raise MSGErrorCallSignDe

    if icq or pack_basecall(call_to) < 0:
        # CQ with non-std call, should use free text (without hash)
        raise MSGErrorCallSignTo

    if not icq:
        # choose which of the callsigns to encode as plain-text (58 bits) or hash (12 bits)
        iflip = call_de.startswith("<") and call_de.endswith(">")  # call_de will be sent plain-text

        call12 = call_de if iflip else call_to
        call58 = call_to if iflip else call_de

        if (x := save_callsign(call12)) is None:
            raise MSGErrorCallSignTo

        _, n12, _ = x
    else:
        iflip = False
        n12 = 0
        call58 = call_de

    if (n58 := pack58(call58)) is None:
        raise MSGErrorCallSignDe

    if icq:
        nrpt = 0
    else:
        nrpt = FTX_EXTRAS_CODE.get(extra, 0)

    # Pack into 12 + 58 + 1 + 2 + 1 + 3 == 77 bits
    # write(c77,1010) n12,n58,iflip,nrpt,icq,i3
    # format(b12.12,b58.58,b1,b2.2,b1,b3.3)
    payload = bytearray(b"\x00" * 10)
    payload[0] = byte(n12 >> 4)
    payload[1] = byte(n12 << 4) | byte(n58 >> 54)
    payload[2] = byte(n58 >> 46)
    payload[3] = byte(n58 >> 38)
    payload[4] = byte(n58 >> 30)
    payload[5] = byte(n58 >> 22)
    payload[6] = byte(n58 >> 14)
    payload[7] = byte(n58 >> 6)
    payload[8] = byte(n58 << 2) | byte(int(iflip) << 1) | byte(nrpt >> 1)
    payload[9] = byte(nrpt << 7) | byte(int(icq) << 6) | byte(i3 << 3)

    return payload


def message_encode_free(text: str) -> typing.ByteString:
    if len(text) > MSG_MESSAGE_FREE_TEXT_LEN:
        raise MSGErrorTooLong

    payload = bytearray(b"\x00" * MSG_MESSAGE_TELEMETRY_LEN)
    text = (" " * (MSG_MESSAGE_FREE_TEXT_LEN - len(text))) + text
    for c in text:
        if (cid := nchar(c, FTX_CHAR_TABLE_FULL)) == -1:
            raise MSGErrorInvalidChar

        rem = cid
        for i in reversed(range(MSG_MESSAGE_TELEMETRY_LEN)):
            rem += payload[i] * len(FTX_CHAR_TABLE_FULL)
            payload[i] = byte(rem)
            rem >>= 8

    return message_encode_telemetry(payload)


def message_encode_telemetry(payload: typing.ByteString) -> typing.ByteString:
    if len(payload) > MSG_MESSAGE_TELEMETRY_LEN:
        raise MSGErrorTooLong

    # Shift bits in payload right by 1 bit to right-align the data
    carry = 0
    data = bytearray(b"\x00" * len(payload))
    for i, t_byte in enumerate(reversed(payload)):
        data[-i - 1] = byte((carry >> 7) | (t_byte << 1))
        carry = byte(t_byte & 0x80)

    return data


def message_decode_telemetry(data: typing.ByteString) -> typing.Generator[int, None, None]:
    # Shift bits in payload right by 1 bit to right-align the data
    carry = 0
    for p_byte in data:
        yield byte((carry << 7) | (p_byte >> 1))
        carry = byte(p_byte & 0x01)


def message_decode_telemetry_hex(data: typing.ByteString) -> str:
    b71 = message_decode_telemetry(data)
    return "".join(f"{b:x}" for b in b71)


def message_decode_free(data: typing.ByteString) -> str:
    payload = bytearray(message_decode_telemetry(data))
    text = " "
    for _ in range(MSG_MESSAGE_FREE_TEXT_LEN):
        # Divide the long integer in payload by 42
        rem = 0
        for i in range(MSG_MESSAGE_TELEMETRY_LEN):
            rem = (rem << 8) | payload[i]
            payload[i] = byte(rem // len(FTX_CHAR_TABLE_FULL))
            rem = rem % len(FTX_CHAR_TABLE_FULL)

        text = charn(rem, FTX_CHAR_TABLE_FULL) + text

    return text.strip()
