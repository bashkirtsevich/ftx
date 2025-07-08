import typing
from contextlib import suppress

from consts import FTX_CALLSIGN_HASH_12_BITS
from consts import FTX_EXTRAS_CODE
from consts import FTX_EXTRAS_STR
from consts import FTX_MESSAGE_TYPE_ARRL_FD
from consts import FTX_MESSAGE_TYPE_ARRL_RTTY
from consts import FTX_MESSAGE_TYPE_DXPEDITION
from consts import FTX_MESSAGE_TYPE_EU_VHF
from consts import FTX_MESSAGE_TYPE_FREE_TEXT
from consts import FTX_MESSAGE_TYPE_NONSTD_CALL
from consts import FTX_MESSAGE_TYPE_STANDARD
from consts import FTX_MESSAGE_TYPE_TELEMETRY
from consts import FTX_MESSAGE_TYPE_UNKNOWN
from consts import FTX_MESSAGE_TYPE_WWROF
from exceptions import FTXErrorCallSignTo, FTXErrorTooLong, FTXErrorInvalidChar, FTXException
from exceptions import FTXErrorCallSignDe
from exceptions import FTXErrorGrid
from exceptions import FTXErrorMsgType
from exceptions import FTXErrorSuffix
from pack import pack_callsign, save_callsign, pack_extra, pack58, unpack_callsign, unpack_extra, lookup_callsign, \
    unpack58, \
    pack_basecall
from text import FTX_CHAR_TABLE_FULL, charn, nchar, endswith_any
from tools import byte, dword


def ftx_message_encode(call_to: str, call_de: str, extra: str = "") -> typing.ByteString:
    if len(call_to) > 11:
        raise FTXErrorCallSignTo

    if len(call_de) > 11:
        raise FTXErrorCallSignDe

    if len(extra) > 19:
        raise FTXErrorGrid

    with suppress(FTXException):
        return ftx_message_encode_std(call_to, call_de, extra)

    with suppress(FTXException):
        return ftx_message_encode_nonstd(call_to, call_de, extra)

    with suppress(FTXException):
        return ftx_message_encode_free(call_to)


def ftx_message_decode(payload: typing.ByteString) -> typing.Tuple[str, typing.Optional[str], typing.Optional[str]]:
    msg_type = ftx_message_get_type(payload)
    if msg_type == FTX_MESSAGE_TYPE_STANDARD:
        field1, field2, field3 = ftx_message_decode_std(payload)
    elif msg_type == FTX_MESSAGE_TYPE_NONSTD_CALL:
        field1, field2, field3 = ftx_message_decode_nonstd(payload)
    elif msg_type == FTX_MESSAGE_TYPE_FREE_TEXT:
        field1 = ftx_message_decode_free(payload)
        field2 = None
        field3 = None
    elif msg_type == FTX_MESSAGE_TYPE_TELEMETRY:
        field1 = ftx_message_decode_telemetry_hex(payload)
        field2 = None
        field3 = None
    else:
        # not handled yet
        raise FTXErrorMsgType

    return field1, field2, field3


def ftx_message_get_type(payload: typing.ByteString) -> int:
    # Extract i3 (bits 74..76)
    # FIXME: Optimize, use dict instead
    i3 = (payload[9] >> 3) & 0x07
    if i3 == 0:
        # Extract n3 (bits 71..73)
        n3 = ((payload[8] << 2) & 0x04) | ((payload[9] >> 6) & 0x03)
        if n3 == 0:
            return FTX_MESSAGE_TYPE_FREE_TEXT
        elif n3 == 1:
            return FTX_MESSAGE_TYPE_DXPEDITION
        elif n3 == 2:
            return FTX_MESSAGE_TYPE_EU_VHF
        elif n3 < 5:
            return FTX_MESSAGE_TYPE_ARRL_FD
        elif n3 == 5:
            return FTX_MESSAGE_TYPE_TELEMETRY
        else:
            return FTX_MESSAGE_TYPE_UNKNOWN
    elif i3 < 3:
        return FTX_MESSAGE_TYPE_STANDARD
    elif i3 == 3:
        return FTX_MESSAGE_TYPE_ARRL_RTTY
    elif i3 == 4:
        return FTX_MESSAGE_TYPE_NONSTD_CALL
    elif i3 == 5:
        return FTX_MESSAGE_TYPE_WWROF
    else:
        return FTX_MESSAGE_TYPE_UNKNOWN


def ftx_message_encode_std(call_to: str, call_de: str, extra: str) -> typing.ByteString:
    b28_to, sh_to = pack_callsign(call_to)
    if b28_to < 0:
        raise FTXErrorCallSignTo

    b28_de, sh_de = pack_callsign(call_de)
    if b28_de < 0:
        raise FTXErrorCallSignDe

    suffix = 1  # No suffix or /R
    if any(cs.endswith("/P") for cs in (call_to, call_de)):
        suffix = 2  # Suffix /P for EU VHF contest
        if any(cs.endswith("/R") for cs in (call_to, call_de)):
            raise FTXErrorSuffix

    if call_to == "CQ" and "/" in call_de and not endswith_any(call_de, "/P", "/R"):
        raise FTXErrorCallSignDe  # nonstandard call: need a type 4 message

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


def ftx_message_decode_std(payload: typing.ByteString) -> typing.Tuple[str, str, str]:
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
        raise FTXErrorCallSignTo

    if (call_de := unpack_callsign(b29_de >> 1, bool(b29_de & 1), cs_flags)) is None:
        raise FTXErrorCallSignDe

    if (extra := unpack_extra(b16_extra, bool(r_flag & 1))) is None:
        raise FTXErrorGrid

    return call_to, call_de, extra


def ftx_message_decode_nonstd(payload: typing.ByteString) -> typing.Tuple[str, str, str]:
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
    call_3 = lookup_callsign(FTX_CALLSIGN_HASH_12_BITS, n12)

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


def ftx_message_encode_nonstd(call_to: str, call_de: str, extra: str) -> typing.ByteString:
    i3 = 4

    icq = call_to == "CQ"
    len_call_to = len(call_to)
    len_call_de = len(call_de)

    if not icq and len_call_to < 3:
        raise FTXErrorCallSignTo

    if len_call_de < 3:
        raise FTXErrorCallSignDe

    if icq or pack_basecall(call_to) < 0:
        # CQ with non-std call, should use free text (without hash)
        raise FTXErrorCallSignTo

    if not icq:
        # choose which of the callsigns to encode as plain-text (58 bits) or hash (12 bits)
        iflip = call_de.startswith("<") and call_de.endswith(">")  # call_de will be sent plain-text

        call12 = call_de if iflip else call_to
        call58 = call_to if iflip else call_de

        if (x := save_callsign(call12)) is None:
            raise FTXErrorCallSignTo

        _, n12, _ = x
    else:
        iflip = False
        n12 = 0
        call58 = call_de

    if (n58 := pack58(call58)) is None:
        raise FTXErrorCallSignDe

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


def ftx_message_encode_free(text: str) -> typing.ByteString:
    if len(text) > 12:
        raise FTXErrorTooLong

    b71 = bytearray(b"\x00" * 12)
    text = (" " * (12 - len(text))) + text
    for c in text:
        if (cid := nchar(c, FTX_CHAR_TABLE_FULL)) == -1:
            raise FTXErrorInvalidChar

        rem = cid
        for i in reversed(range(9)):
            rem += b71[i] * 42
            b71[i] = byte(rem)
            rem >>= 8

    return ftx_message_encode_telemetry(b71)


def ftx_message_encode_telemetry(telemetry: typing.ByteString) -> typing.ByteString:
    # Shift bits in payload right by 1 bit to right-align the data
    carry = 0
    data = bytearray(b"\x00" * len(telemetry))
    for i, t_byte in enumerate(reversed(telemetry)):
        data[-i - 1] = byte((carry >> 7) | (t_byte << 1))
        carry = byte(t_byte & 0x80)

    return data


def ftx_message_decode_telemetry(payload: typing.ByteString) -> typing.Generator[int, None, None]:
    # Shift bits in payload right by 1 bit to right-align the data
    carry = 0
    for p_byte in payload:
        yield byte((carry << 7) | (p_byte >> 1))
        carry = byte(p_byte & 0x01)


def ftx_message_decode_telemetry_hex(payload: typing.ByteString) -> str:
    b71 = ftx_message_decode_telemetry(payload)
    return "".join(f"{b:x}" for b in b71)


def ftx_message_decode_free(payload: typing.ByteString) -> str:
    b71 = bytearray(ftx_message_decode_telemetry(payload))
    c14 = " "
    for _ in range(12):
        # Divide the long integer in b71 by 42
        rem = 0
        for i in range(9):
            rem = (rem << 8) | b71[i]
            b71[i] = byte(rem // 42)
            rem = rem % 42

        c14 = charn(rem, FTX_CHAR_TABLE_FULL) + c14

    return c14.strip()
