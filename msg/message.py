import re
import typing
from abc import ABCMeta, abstractmethod

from consts.msg import MSG_MESSAGE_TYPE_FREE_TEXT
from consts.msg import MSG_MESSAGE_TYPE_DXPEDITION
from consts.msg import MSG_MESSAGE_TYPE_EU_VHF
from consts.msg import MSG_MESSAGE_TYPE_ARRL_FD
from consts.msg import MSG_MESSAGE_TYPE_TELEMETRY
from consts.msg import MSG_MESSAGE_TYPE_STANDARD
from consts.msg import MSG_MESSAGE_TYPE_ARRL_RTTY
from consts.msg import MSG_MESSAGE_TYPE_NONSTD_CALL
from consts.msg import MSG_MESSAGE_TYPE_WWROF
from consts.msg import MSG_MESSAGE_TYPE_UNKNOWN
from consts.msg import MSG_MESSAGE_FREE_TEXT_LEN
from consts.msg import MSG_MESSAGE_TELEMETRY_LEN
from consts.msg import MSG_NTOKENS
from consts.msg import MSG_MAX_22

from consts.ftx import FTX_EXTRAS_CODE
from consts.ftx import FTX_MAX_GRID_4
from consts.ftx import FTX_TOKEN_STR
from consts.ftx import FTX_TOKEN_CODE
from consts.ftx import FTX_RESPONSE_EXTRAS_CODE
from consts.ftx import FTX_RESPONSE_EXTRAS_STR
from consts.ftx import FTX_EXTRAS_STR

from .exceptions import MSGErrorTooLong
from .exceptions import MSGErrorInvalidChar
from .exceptions import MSGNotImplemented
from .exceptions import MSGInvalidCallsign

from .text import FTX_CHAR_TABLE_FULL
from .text import FTX_CHAR_TABLE_ALPHANUM_SPACE_SLASH
from .text import FTX_GRID_CHAR_MAP
from .text import FTX_CHAR_TABLE_LETTERS_SPACE
from .text import FTX_BASECALL_CHAR_MAP
from .text import charn
from .text import nchar

from .text import ct_encode
from .text import ct_validate
from .text import ct_map_encode

from .text import ct_map_decode
from .text import ct_decode
from .text import ct_validate_map

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

    def __int__(self):
        return self.val_int

    def __repr__(self):
        return str(self)


class BaseCallsign(MsgItem):
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


class DummyCallsign(BaseCallsign):
    @classmethod
    def _validate_str(cls, val: str) -> bool:
        return True

    @classmethod
    def _validate_int(cls, val: int) -> bool:
        return True

    def __init__(self):
        super().__init__(-1)

    def to_int(self) -> int:
        return -1

    def to_str(self) -> str:
        return "<...>"

    def hash_22(self):
        return -1

    def hash_12(self):
        return -1

    def hash_10(self):
        return -1


_DummyCallsign = DummyCallsign()


class Callsign(BaseCallsign):
    @classmethod
    def _validate_str(cls, val: str) -> bool:
        return True

    @classmethod
    def _validate_int(cls, val: int) -> bool:
        return True

    def to_int(self) -> int:
        if self.val_str.startswith("CQ_"):
            return self._pack_cq_call(self.val_str[3:])

        if len(self.val_str) > 2:
            val = self._pack_basecall(self.val_str)
            return MSG_NTOKENS + MSG_MAX_22 + val

        return MSG_NTOKENS + self.hash_22()

    @staticmethod
    def _pack_cq_call(cs: str) -> int:
        if not (1 <= (cs_len := len(cs)) <= 4):
            raise ValueError("Invalid callsign")

        if cs_len == 3 and cs.isdigit():
            return int(cs) + 3

        ct_validate(FTX_CHAR_TABLE_LETTERS_SPACE, cs, raise_exception=True)

        return ct_encode(FTX_CHAR_TABLE_LETTERS_SPACE, cs) + 1003

    @classmethod
    def _pack_basecall(cls, cs: str) -> int:
        cs_norm = cls._normalize_cs(cs)
        return ct_map_encode(FTX_BASECALL_CHAR_MAP, cs_norm)

    @staticmethod
    def _normalize_cs(cs: str) -> str:
        val_len = len(cs)
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
        return cs_norm

    @staticmethod
    def _denormalize_cs(cs: str) -> str:
        # Copy cs to 6 character buffer
        if cs.startswith("3D0") and cs[3] != " ":
            # Work-around for Swaziland prefix: 3D0XYZ -> 3DA0XYZ
            cs = f"3DA0{cs[3:]}"
        elif cs[0] == "Q" and cs[1].isalpha():
            # Work-around for Guinea prefixes: QA0XYZ -> 3XA0XYZ
            cs = f"3X{cs[1:]}"
        # Skip trailing and leading whitespace in case of a short cs
        return cs.strip()

    def to_str(self) -> str:
        # Check for special tokens DE, QRZ, CQ, CQ_nnn, CQ_aaaa
        val = self.val_int
        if val < MSG_NTOKENS:
            if val <= 2:
                raise ValueError("Invalid cs representation")

            if val <= 1002:
                # CQ nnn with 3 digits
                return f"CQ_{val - 3:03}"

            if val <= 532443:
                # CQ ABCD with 4 alphanumeric symbols
                cq_tail = ct_decode(FTX_CHAR_TABLE_LETTERS_SPACE, val - 1003, l=4)
                return f"CQ_{cq_tail.strip()}"

            # unspecified
            raise ValueError("Invalid cs specification")

        val -= MSG_NTOKENS
        if val < MSG_MAX_22:
            raise ValueError("Invalid cs representation")

        # Standard cs
        cs = ct_map_decode(FTX_BASECALL_CHAR_MAP, val - MSG_MAX_22)
        return self._denormalize_cs(cs)


class CallsignExt(BaseCallsign):
    @classmethod
    def _validate_str(cls, val: str) -> bool:
        return len(val) <= 10 and all(c in FTX_CHAR_TABLE_ALPHANUM_SPACE_SLASH for c in val)

    @classmethod
    def _validate_int(cls, val: int) -> bool:
        return True

    def to_int(self) -> int:
        val = "".join(c for c in self.val_str if c not in "<>")
        return ct_encode(FTX_CHAR_TABLE_ALPHANUM_SPACE_SLASH, val)

    def to_str(self) -> str:
        val = ct_decode(FTX_CHAR_TABLE_ALPHANUM_SPACE_SLASH, self.val_int, 10)
        return val.strip()


class Grid(MsgItem):
    @classmethod
    def _validate_str(cls, val: str) -> bool:
        return len(val) == 4 and ct_validate_map(FTX_GRID_CHAR_MAP, val)

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


class TokenDE(Token):
    def __init__(self):
        super().__init__("DE")


class TokenQRZ(Token):
    def __init__(self):
        super().__init__("QRZ")


class TokenCQ(Token):
    def __init__(self):
        super().__init__("CQ")


class Extra(_DictItem):
    int_dict = FTX_EXTRAS_CODE
    str_dict = FTX_EXTRAS_STR


class ExtraRRR(Extra):
    def __init__(self):
        super().__init__("RRR")


class ExtraRR73(Extra):
    def __init__(self):
        super().__init__("RR73")


class Extra73(Extra):
    def __init__(self):
        super().__init__("73")


class ResponseExtra(_DictItem):
    int_dict = FTX_RESPONSE_EXTRAS_CODE
    str_dict = FTX_RESPONSE_EXTRAS_STR


class ResponseExtraEmpty(ResponseExtra):
    def __init__(self):
        super().__init__("")


class ResponseExtraRRR(ResponseExtra):
    def __init__(self):
        super().__init__("RRR")


class ResponseExtraRR73(ResponseExtra):
    def __init__(self):
        super().__init__("RR73")


class ResponseExtra73(ResponseExtra):
    def __init__(self):
        super().__init__("73")


class AbstractMessage(metaclass=ABCMeta):
    def __repr__(self):
        return str(self)

    @abstractmethod
    def encode(self, **kwargs) -> typing.ByteString:
        ...

    @classmethod
    @abstractmethod
    def decode(cls, payload: typing.ByteString, **kwargs) -> "AbstractMessage":
        ...


class StdMessage(AbstractMessage):
    __slots__ = ("to", "de", "extra")

    def __init__(self, to: typing.Union[Token, BaseCallsign], de: BaseCallsign,
                 extra: typing.Union[Grid, Report, Extra, ResponseExtra]):
        self.to = to
        self.de = de
        self.extra = extra

    def __str__(self):
        return f"{self.to} {self.de} {self.extra}"

    @staticmethod
    def _decode_callsign(val: int, msg_server: typing.Optional["MsgServer"] = None):
        if Token.validate(val):
            return Token(val)

        if val < MSG_NTOKENS + MSG_MAX_22:
            if msg_server:
                return msg_server._get_cs(val)
            return _DummyCallsign

        if Callsign.validate(val):
            cs = Callsign(val)
            if msg_server:
                return msg_server._save_cs(cs)
            return cs

        raise MSGInvalidCallsign

    @staticmethod
    def _decode_extra(val: int):
        if Grid.validate(val):
            return Grid(val)

        if ResponseExtra.validate(val):
            return ResponseExtra(val)

        if Report.validate(val):
            return Report(val)

        raise ValueError

    def encode(self, **kwargs) -> typing.ByteString:
        sh_to = 0  # TODO
        sh_de = 0  # TODO

        b29_to = dword(int(self.to) << 1 | sh_to)
        b29_de = dword(int(self.de) << 1 | sh_de)
        b16_extra = int(self.extra)

        suffix = 1  # TODO

        # Pack into (28 + 1) + (28 + 1) + (1 + 15) + 3 bits
        items = [
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
        return bytearray(b for b in items)

    @classmethod
    def decode(cls, payload: typing.ByteString, **kwargs) -> AbstractMessage:
        msg_server = kwargs.get("msg_server")

        # Extract packed fields
        b29_to = payload[0] << 21
        b29_to |= payload[1] << 13
        b29_to |= payload[2] << 5
        b29_to |= payload[3] >> 3
        b29_to >>= 1
        call_to = cls._decode_callsign(b29_to, msg_server)

        b29_de = (payload[3] & 0x07) << 26
        b29_de |= payload[4] << 18
        b29_de |= payload[5] << 10
        b29_de |= payload[6] << 2
        b29_de |= payload[7] >> 6
        b29_de >>= 1
        call_de = cls._decode_callsign(b29_de, msg_server)

        # r_flag = (payload[7] & 0x20) >> 5

        b16_extra = (payload[7] & 0x1F) << 10
        b16_extra |= payload[8] << 2
        b16_extra |= payload[9] >> 6
        extra = cls._decode_extra(b16_extra)

        # Extract cs_flags (bits 74..76)
        # cs_flags = (payload[9] >> 3) & 0x07
        return cls(call_to, call_de, extra)


class NonStdMessage(StdMessage):
    def encode(self, **kwargs) -> typing.ByteString:
        msg_server = kwargs.get("msg_server")

        is_cq = isinstance(self.to, TokenCQ)

        if is_cq:
            flip = False
            n12 = 0
            call_58 = self.de
            xtra = 0
        else:
            # choose which of the callsigns to encode as plain-text (58 bits) or hash (12 bits)
            flip = isinstance(self.de, CallsignExt)

            call_12 = self.de if flip else self.to
            call_58 = self.to if flip else self.de

            n12 = call_12.hash_12()
            if msg_server:
                msg_server._save_cs(call_12)

            xtra = int(self.extra)

        n58 = int(call_58)

        # Pack into 12 + 58 + 1 + 2 + 1 + 3 == 77 bits
        i3 = 4
        items = [
            byte(n12 >> 4),
            byte(n12 << 4) | byte(n58 >> 54),

            byte(n58 >> 46),
            byte(n58 >> 38),
            byte(n58 >> 30),
            byte(n58 >> 22),
            byte(n58 >> 14),
            byte(n58 >> 6),
            byte(n58 << 2) | byte(int(flip) << 1) | byte(xtra >> 1),

            byte(xtra << 7) | byte(int(is_cq) << 6) | byte(i3 << 3),
        ]

        return bytearray(b for b in items)

    @classmethod
    def decode(cls, payload: typing.ByteString, **kwargs) -> AbstractMessage:
        msg_server = kwargs.get("msg_server")
        # non-standard messages, code originally by KD8CEC
        # Decode the other call from hash lookup table
        hash_12 = payload[0] << 4  # 11 ~ 4 : 8
        hash_12 |= payload[1] >> 4  # 3 ~ 0  : 12
        if msg_server:
            cs_3 = msg_server._get_cs(hash_12)
        else:
            cs_3 = _DummyCallsign

        # Decode one of the calls from 58 bit encoded string
        b58_cs = (payload[1] & 0x0F) << 54  # 57 ~ 54 : 4
        b58_cs |= payload[2] << 46  # 53 ~ 46 : 12
        b58_cs |= payload[3] << 38  # 45 ~ 38 : 12
        b58_cs |= payload[4] << 30  # 37 ~ 30 : 12
        b58_cs |= payload[5] << 22  # 29 ~ 22 : 12
        b58_cs |= payload[6] << 14  # 21 ~ 14 : 12
        b58_cs |= payload[7] << 6  # 13 ~ 6  : 12
        b58_cs |= payload[8] >> 2  # 5 ~ 0   : 765432 10
        cs_decoded = CallsignExt(b58_cs)

        if msg_server:
            msg_server._save_cs(cs_decoded)

        # Possibly flip them around
        flag_flip = bool((payload[8] >> 1) & 0x01)  # 76543210
        cs_1 = cs_decoded if flag_flip else cs_3
        cs_2 = cs_3 if flag_flip else cs_decoded

        flag_cq = bool((payload[9] >> 6) & 0x01)
        if flag_cq:
            call_to = TokenCQ()
            extra = ResponseExtraEmpty()
        else:
            call_to = cs_1

            b2_xtra = (payload[8] & 0x01) << 1
            b2_xtra |= payload[9] >> 7  # 76543210
            extra = Extra(b2_xtra)

        call_de = cs_2

        return cls(call_to, call_de, extra)


class Telemetry(AbstractMessage):
    __slots__ = ("data")

    def __init__(self, data: typing.ByteString):
        if len(data) > MSG_MESSAGE_TELEMETRY_LEN:
            raise MSGErrorTooLong("Maximum data length exceeded")

        self.data = data

    @staticmethod
    def _decode_bytes(data) -> typing.Generator[int, None, None]:
        # Shift bits in payload right by 1 bit to right-align the data
        carry = 0
        for p_byte in data:
            yield byte((carry << 7) | (p_byte >> 1))
            carry = byte(p_byte & 0x01)

    def encode(self, **kwargs) -> typing.ByteString:
        # Shift bits in payload right by 1 bit to right-align the data
        carry = 0
        data = bytearray(b"\x00" * len(self.data))
        for i, t_byte in enumerate(reversed(self.data)):
            data[-i - 1] = byte((carry >> 7) | (t_byte << 1))
            carry = byte(t_byte & 0x80)

        return data

    @classmethod
    def decode(cls, payload: typing.ByteString, **kwargs) -> AbstractMessage:
        data = bytearray(cls._decode_bytes(payload))
        return cls(data)

    @property
    def as_hex(self):
        return str(self)

    def __str__(self):
        return "".join(f"{b:02x}" for b in self.data)


class FreeText(Telemetry):
    def __init__(self, text: str):
        if len(text) > MSG_MESSAGE_FREE_TEXT_LEN:
            raise MSGErrorTooLong("Maximum text length exceeded")

        data = self._encode_str(text)
        super().__init__(data)

    @staticmethod
    def _encode_str(text: str) -> typing.ByteString:
        data = bytearray(b"\x00" * MSG_MESSAGE_TELEMETRY_LEN)
        text = (" " * (MSG_MESSAGE_FREE_TEXT_LEN - len(text))) + text
        for c in text:
            if (c_id := nchar(c, FTX_CHAR_TABLE_FULL)) == -1:
                raise MSGErrorInvalidChar

            rem = c_id
            for i in reversed(range(MSG_MESSAGE_TELEMETRY_LEN)):
                rem += data[i] * len(FTX_CHAR_TABLE_FULL)
                data[i] = byte(rem)
                rem >>= 8

        return data

    @staticmethod
    def _decode_str(data: typing.ByteString) -> str:
        ct = FTX_CHAR_TABLE_FULL
        ct_len = len(ct)

        text = ""
        for _ in range(MSG_MESSAGE_FREE_TEXT_LEN):
            # Divide the long integer in payload by 42
            rem = 0
            for i in range(MSG_MESSAGE_TELEMETRY_LEN):
                rem = (rem << 8) | data[i]
                data[i] = byte(rem // ct_len)
                rem = rem % ct_len

            text = charn(rem, ct) + text

        return text.strip()

    @classmethod
    def decode(cls, payload: typing.ByteString, **kwargs) -> AbstractMessage:
        data = bytearray(cls._decode_bytes(payload))
        text = cls._decode_str(data)
        return cls(text)

    @property
    def as_str(self):
        return str(self)

    def __str__(self):
        data = self.data[:]
        return self._decode_str(data)


class MsgServer:
    __slots__ = ("callsigns",)

    MSG_CLASSES = {
        MSG_MESSAGE_TYPE_STANDARD: StdMessage,
        MSG_MESSAGE_TYPE_NONSTD_CALL: NonStdMessage,
        MSG_MESSAGE_TYPE_FREE_TEXT: FreeText,
        MSG_MESSAGE_TYPE_TELEMETRY: Telemetry,
    }

    def __init__(self):
        self.callsigns = dict()

    def _get_cs(self, cs_hash: int) -> BaseCallsign:
        return self.callsigns.get(cs_hash, _DummyCallsign)

    def _save_cs(self, callsign: BaseCallsign) -> BaseCallsign:
        hashes = [callsign.hash_22(),
                  callsign.hash_12(),
                  callsign.hash_10()]

        for h in hashes:
            self.callsigns[h] = callsign

        return callsign

    def decode(self, payload: typing.ByteString) -> AbstractMessage:
        msg_type = self._msg_get_type(payload)
        if msg_class := self.MSG_CLASSES.get(msg_type):
            return msg_class.decode(payload, msg_server=self)

        raise MSGNotImplemented(f"Unsupported msg type {msg_type}")

    @staticmethod
    def _msg_get_type(payload: typing.ByteString) -> int:
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
