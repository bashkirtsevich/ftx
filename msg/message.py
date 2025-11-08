import re
import typing
from abc import ABCMeta, abstractmethod
from contextlib import suppress

from consts.msg import MSG_CALLSIGN_HASH_12_BITS, MSG_MESSAGE_TYPE_FREE_TEXT, MSG_MESSAGE_TYPE_DXPEDITION, \
    MSG_MESSAGE_TYPE_EU_VHF, MSG_MESSAGE_TYPE_ARRL_FD, MSG_MESSAGE_TYPE_TELEMETRY, MSG_MESSAGE_TYPE_STANDARD, \
    MSG_MESSAGE_TYPE_ARRL_RTTY, MSG_MESSAGE_TYPE_NONSTD_CALL, MSG_MESSAGE_TYPE_WWROF, MSG_MESSAGE_TYPE_UNKNOWN, \
    MSG_MESSAGE_FREE_TEXT_LEN, MSG_MESSAGE_TELEMETRY_LEN
from consts.ftx import FTX_EXTRAS_CODE, FTX_MAX_GRID_4, FTX_TOKEN_STR, FTX_TOKEN_CODE, FTX_RESPONSE_EXTRAS_CODE, \
    FTX_RESPONSE_EXTRAS_STR
from consts.ftx import FTX_EXTRAS_STR
from .exceptions import MSGErrorCallSignTo, MSGErrorTooLong, MSGErrorInvalidChar, MSGException
from .exceptions import MSGErrorCallSignDe
from .exceptions import MSGErrorGrid
from .exceptions import MSGErrorMsgType
from .exceptions import MSGErrorSuffix
from .pack import pack_callsign, save_callsign, pack_extra, pack58, unpack_callsign, unpack_extra, lookup_callsign, \
    unpack58, \
    pack_basecall, pack_grid, MAX22, NTOKENS
from .text import FTX_CHAR_TABLE_FULL, charn, nchar, endswith_any, FTX_CHAR_TABLE_ALPHANUM_SPACE_SLASH, \
    FTX_GRID_CHAR_MAP, FTX_CHAR_TABLE_LETTERS_SPACE, ct_encode, ct_validate, ct_map_encode, FTX_BASECALL_CHAR_MAP, \
    ct_map_decode, ct_decode
from .tools import byte, dword


class MsgItem(metaclass=ABCMeta):
    __slots__ = ("val_str", "val_int")

    def __init__(self, val: typing.Union[str, int]):
        if not self.validate(val):
            raise ValueError("Validation error")

        if isinstance(val, str):
            self.val_str = val.strip()
            self.val_int = self.to_int()
        elif isinstance(val, int):
            self.val_int = val
            self.val_str = self.to_str()
        else:
            raise TypeError(f"Unsupported data type {type(val)}")

    @classmethod
    @abstractmethod
    def _validate_str(cls, val: str) -> bool:
        ...

    @classmethod
    @abstractmethod
    def _validate_int(cls, val: int) -> bool:
        ...

    @classmethod
    def _validate_error(cls, msg: str):
        raise ValueError(msg)

    @classmethod
    def validate(cls, val: typing.Union[str, int], raise_exception: bool = False) -> bool:
        # TODO: Add raise_exception logic
        if isinstance(val, str):
            ok = cls._validate_str(val)
        elif isinstance(val, int):
            ok = cls._validate_int(val)
        else:
            raise TypeError("Unsupported value type")

        return ok

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


class Callsign(MsgItem):
    @classmethod
    def _validate_str(cls, val: str) -> bool:
        return True

    @classmethod
    def _validate_int(cls, val: int) -> bool:
        return True

    def to_int(self) -> int:
        if self.val_str.startswith("CQ_"):
            return self._pack_cq_call(self.val_str[3:])

        if val := self._pack_basecall(self.val_str):
            return NTOKENS + MAX22 + val

        return NTOKENS + self.hash_22()

    @staticmethod
    def _pack_cq_call(cs: str) -> int:
        if not (1 <= (cs_len := len(cs)) <= 4):
            raise ValueError("Invalid callsign")

        if cs_len == 3 and cs.isdigit():
            return int(cs) + 3

        ct_validate(FTX_CHAR_TABLE_LETTERS_SPACE, cs, raise_exception=True)

        return ct_encode(FTX_CHAR_TABLE_LETTERS_SPACE, cs) + 1003

    @staticmethod
    def _pack_basecall(cs: str) -> typing.Optional[int]:
        if (val_len := len(cs)) <= 2:
            return None

        # Work-around for Swaziland prefix: 3DA0XYZ -> 3D0XYZ
        if cs.startswith("3DA0") and 4 < val_len <= 7:
            cs_norm = f"3D0{cs[4:]}"
        # Work-around for Guinea prefixes: 3XA0XYZ -> QA0XYZ
        elif cs.startswith("3X") and cs[2].isalpha() and val_len <= 7:
            cs_norm = f"Q{cs[2:]}"
        elif cs[2].isdigit() and val_len <= 6:
            cs_norm = cs
        # Check the position of callsign digit and make a right-aligned copy into cs_norm
        elif cs[1].isdigit() and val_len <= 5:
            # A0XYZ -> " A0XYZ"
            cs_norm = f" {cs}"
        else:
            cs_norm = ""

        cs_norm += " " * (6 - len(cs_norm))  # Normalize to 6 letters

        return ct_map_encode(FTX_BASECALL_CHAR_MAP, cs_norm)

    def to_str(self) -> str:
        # Check for special tokens DE, QRZ, CQ, CQ_nnn, CQ_aaaa
        val = self.val_int
        if val < NTOKENS:
            if val <= 2:
                raise ValueError("Invalid cs representation")

            if val <= 1002:
                # CQ nnn with 3 digits
                return f"CQ_{val - 3:03}"

            if val <= 532443:
                # CQ ABCD with 4 alphanumeric symbols
                aaaa = ct_decode(FTX_CHAR_TABLE_LETTERS_SPACE, val - 1003, l=4)

                return f"CQ_{aaaa.strip()}"

            # unspecified
            raise ValueError("Invalid cs specification")

        val -= NTOKENS
        if val < MAX22:
            raise ValueError("Invalid cs representation")

        # Standard cs
        cs = ct_map_decode(FTX_BASECALL_CHAR_MAP, val - MAX22)

        # Copy cs to 6 character buffer
        if cs.startswith("3D0") and cs[3] != " ":
            # Work-around for Swaziland prefix: 3D0XYZ -> 3DA0XYZ
            cs = f"3DA0{cs[3:]}"
        elif cs[0] == "Q" and cs[1].isalpha():
            # Work-around for Guinea prefixes: QA0XYZ -> 3XA0XYZ
            cs = f"3X{cs[1:]}"

        # Skip trailing and leading whitespace in case of a short cs
        return cs.strip()

    def hash_22(self):
        ct = FTX_CHAR_TABLE_ALPHANUM_SPACE_SLASH
        val = ct_encode(ct, self.val_str)

        # pretend to have trailing whitespace (with j=0, index of ' ')
        if (val_len := len(self.val_str)) < 11:
            val *= len(ct) ** (11 - val_len)

        hash = ((47055833459 * val) >> (64 - 22)) & 0x3fffff
        return hash

    def hash_12(self):
        hash = self.hash_22()
        return hash >> 10

    def hash_10(self):
        hash = self.hash_22()
        return hash >> 12

    def __hash__(self):
        return self.hash_22()


class Grid(MsgItem):
    @classmethod
    def _validate_str(cls, val: str) -> bool:
        return len(val) == 4 and ct_validate(FTX_GRID_CHAR_MAP, val)

    @classmethod
    def _validate_int(cls, val: int) -> bool:
        return val <= FTX_MAX_GRID_4

    def to_int(self) -> int:
        return ct_map_encode(FTX_GRID_CHAR_MAP, self.val_str)

    def to_str(self) -> str:
        return ct_map_decode(FTX_GRID_CHAR_MAP, self.val_int)


class Report(MsgItem):
    report_regex = re.compile(r"^(R){0,1}([\+\-]{0,1})([0-9]+)$")

    @classmethod
    def _parse_report(cls, val: str) -> typing.Optional[int]:
        if report := cls.report_regex.match(val):
            _, sign, val = report.groups()

            return int(sign + val)

        return None

    @classmethod
    def _validate_str(cls, val: str) -> bool:
        report = cls._parse_report(val)
        return isinstance(report, int) and -35 <= report <= 35

    @classmethod
    def _validate_int(cls, val: int) -> bool:
        return val > FTX_MAX_GRID_4

    def to_int(self) -> int:
        report = self._parse_report(self.val_str)
        return FTX_MAX_GRID_4 + report + 35

    def to_str(self) -> str:
        val = int(self.val_int - FTX_MAX_GRID_4 - 35)
        return f"R{val:+03}"


class _DictItem(MsgItem):
    int_dict = None
    str_dict = None

    @classmethod
    def _validate_str(cls, val: str) -> bool:
        return val in cls.int_dict

    @classmethod
    def _validate_int(cls, val: int) -> bool:
        return val in cls.str_dict

    def to_int(self) -> int:
        return self.int_dict[self.val_str]

    def to_str(self) -> str:
        return self.str_dict[self.val_int]


class Token(_DictItem):
    int_dict = FTX_TOKEN_CODE
    str_dict = FTX_TOKEN_STR


class Extra(_DictItem):
    int_dict = FTX_EXTRAS_CODE
    str_dict = FTX_EXTRAS_STR


class ResponseExtra(_DictItem):
    int_dict = FTX_RESPONSE_EXTRAS_CODE
    str_dict = FTX_RESPONSE_EXTRAS_STR


class AbstractMessage:
    def __repr__(self):
        return str(self)


class StdMessage(AbstractMessage):
    __slots__ = ("to", "de", "extra")

    def __init__(self, to: typing.Union[Token, Callsign], de: Callsign, extra: typing.Union[Grid, Report, Extra]):
        self.to = to
        self.de = de
        self.extra = extra

    def __str__(self):
        return f"{self.to} {self.de} {self.extra}"


class MsgServer:
    __slots__ = ("callsigns")

    def __init__(self):
        self.callsigns = dict()

    def decode(self, payload: typing.ByteString) -> AbstractMessage:
        msg_type = message_get_type(payload)

        if msg_type == MSG_MESSAGE_TYPE_STANDARD:
            return self._decode_std(payload)

        raise NotImplemented

    def decode_callsign(self, val: int):
        if Token.validate(val):
            return Token(val)

        if val < NTOKENS + MAX22:
            return self.callsigns.get(val, "<...>")

        if Callsign.validate(val):
            cs = Callsign(val)
            self.callsigns[cs.hash_22()] = cs

            return cs

        raise ValueError

    def decode_extra(self, val: int):
        if Grid.validate(val):
            return Grid(val)

        if ResponseExtra.validate(val):
            return ResponseExtra(val)

        if Report.validate(val):
            return Report(val)

        raise ValueError

    def _decode_std(self, payload: typing.ByteString) -> AbstractMessage:
        # Extract packed fields
        b29_to = payload[0] << 21
        b29_to |= payload[1] << 13
        b29_to |= payload[2] << 5
        b29_to |= payload[3] >> 3
        b29_to >>= 1

        b29_de = (payload[3] & 0x07) << 26
        b29_de |= payload[4] << 18
        b29_de |= payload[5] << 10
        b29_de |= payload[6] << 2
        b29_de |= payload[7] >> 6
        b29_de >>= 1

        # r_flag = (payload[7] & 0x20) >> 5

        b16_extra = (payload[7] & 0x1F) << 10
        b16_extra |= payload[8] << 2
        b16_extra |= payload[9] >> 6

        # Extract cs_flags (bits 74..76)
        # cs_flags = (payload[9] >> 3) & 0x07

        call_to = self.decode_callsign(b29_to)
        call_de = self.decode_callsign(b29_de)
        extra = self.decode_extra(b16_extra)

        # cs_flags
        return StdMessage(call_to, call_de, extra)


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
