import typing

from consts import FTX_CALLSIGN_HASH_12_BITS
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
from exceptions import FTXErrorCallSign1
from exceptions import FTXErrorCallSign2
from exceptions import FTXErrorGrid
from exceptions import FTXErrorMsgType
from exceptions import FTXErrorSuffix
from pack import pack28, save_callsign, packgrid, pack58, unpack28, unpackgrid, lookup_callsign, unpack58
from text import FT8_CHAR_TABLE_FULL, charn
from tools import byte, dword


def ftx_message_encode(call_to: str, call_de: str, extra: str = "") -> typing.ByteString:
    if len(call_to) > 11:
        raise FTXErrorCallSign1

    if len(call_de) > 11:
        raise FTXErrorCallSign2

    if len(extra) > 19:
        raise FTXErrorGrid

    try:
        message = ftx_message_encode_std(call_to, call_de, extra)
        return message
    except:
        message = ftx_message_encode_nonstd(call_to, call_de, extra)
        return message


def ftx_message_decode(
        payload: typing.ByteString
) -> typing.Tuple[typing.Optional[str], typing.Optional[str], typing.Optional[str]]:
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
    ipa, n28a = pack28(call_to)
    ipb, n28b = pack28(call_de)

    if n28a < 0:
        raise FTXErrorCallSign1

    if n28b < 0:
        raise FTXErrorCallSign2

    i3 = 1  # No suffix or /R
    if call_to.endswith("/P") or call_de.endswith("/P"):  # FIXME: Use "any(...)"
        i3 = 2  # Suffix /P for EU VHF contest
        if call_to.endswith("/R") or call_de.endswith("/R"):  # FIXME: Use "any(...)"
            raise FTXErrorSuffix

    igrid4 = packgrid(extra)

    # Shift in ipa and ipb bits into n28a and n28b
    n29a = dword(n28a << 1 | ipa)
    n29b = dword(n28b << 1 | ipb)

    # TODO: check for suffixes
    if call_to.endswith("/R"):
        n29a |= 1  # ipa = 1
    elif call_to.endswith("/P"):
        n29a |= 1  # ipa = 1
        i3 = 2

    # Pack into (28 + 1) + (28 + 1) + (1 + 15) + 3 bits
    payload = bytearray(b"\x00" * 10)
    payload[0] = byte(n29a >> 21)
    payload[1] = byte(n29a >> 13)
    payload[2] = byte(n29a >> 5)
    payload[3] = byte(n29a << 3) | byte(n29b >> 26)
    payload[4] = byte(n29b >> 18)
    payload[5] = byte(n29b >> 10)
    payload[6] = byte(n29b >> 2)
    payload[7] = byte(n29b << 6) | byte(igrid4 >> 10)
    payload[8] = byte(igrid4 >> 2)
    payload[9] = byte(igrid4 << 6) | byte(i3 << 3)

    return payload


def ftx_message_decode_std(payload: typing.ByteString) -> typing.Tuple[str, str, str]:
    #  Extract packed fields
    n29a = (payload[0] << 21)
    n29a |= (payload[1] << 13)
    n29a |= (payload[2] << 5)
    n29a |= (payload[3] >> 3)
    n29b = ((payload[3] & 0x07) << 26)
    n29b |= (payload[4] << 18)
    n29b |= (payload[5] << 10)
    n29b |= (payload[6] << 2)
    n29b |= (payload[7] >> 6)

    ir = ((payload[7] & 0x20) >> 5)
    igrid4 = ((payload[7] & 0x1F) << 10)
    igrid4 |= (payload[8] << 2)
    igrid4 |= (payload[9] >> 6)

    # Extract i3 (bits 74..76)
    i3 = (payload[9] >> 3) & 0x07
    # LOG(LOG_DEBUG, "decode_std() n28a=%d ipa=%d n28b=%d ipb=%d ir=%d igrid4=%d i3=%d\n", n29a >> 1, n29a & 1u, n29b >> 1, n29b & 1u, ir, igrid4, i3);

    if (call_to := unpack28(n29a >> 1, n29a & 1, i3)) is None:
        raise FTXErrorCallSign1

    if (call_de := unpack28(n29b >> 1, n29b & 1, i3)) is None:
        raise FTXErrorCallSign2

    if (extra := unpackgrid(igrid4, ir)) is None:
        raise FTXErrorGrid

    return call_to, call_de, extra


# non-standard messages, code originally by KD8CEC
def ftx_message_decode_nonstd(payload: typing.ByteString) -> typing.Tuple[str, str, str]:
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

        extra_dict = {
            1: "RRR",
            2: "RR73",
            3: "73"
        }

        extra = extra_dict.get(nrpt, "")
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
        raise FTXErrorCallSign1

    if len_call_de < 3:
        raise FTXErrorCallSign2

    if not icq:
        # choose which of the callsigns to encode as plain-text (58 bits) or hash (12 bits)
        iflip = call_de.startswith("<") and call_de.endswith(">")  # call_de will be sent plain-text

        call12 = call_de if iflip else call_to
        call58 = call_to if iflip else call_de

        if (x := save_callsign(call12)) is None:
            raise FTXErrorCallSign1

        _, n12, _ = x
    else:
        iflip = False
        n12 = 0
        call58 = call_de

    if (n58 := pack58(call58)) is None:
        raise FTXErrorCallSign2

    if icq:
        nrpt = 0
    else:
        extra_dict = {
            "RRR": 1,
            "RR73": 2,
            "73": 3
        }
        nrpt = extra_dict.get(extra, 0)

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


def ftx_message_decode_telemetry(payload: typing.ByteString) -> typing.ByteString:
    # Shift bits in payload right by 1 bit to right-align the data
    carry = 0
    telemetry = bytearray(b"\x00" * 9)
    for i in range(9):
        telemetry[i] = byte((carry << 7) | (payload[i] >> 1))
        carry = byte(payload[i] & 0x01)

    return telemetry


def ftx_message_decode_telemetry_hex(payload: typing.ByteString) -> str:
    b71 = ftx_message_decode_telemetry(payload)
    return "".join(f"{b:x}" for b in b71)


def ftx_message_decode_free(payload: typing.ByteString) -> str:
    b71 = ftx_message_decode_telemetry(payload)
    c14 = ""
    for idx in range(12):
        # Divide the long integer in b71 by 42
        rem = 0
        for i in range(9):
            rem = (rem << 8) | b71[i]
            b71[i] = rem // 42
            rem = rem % 42

        c14 = charn(rem, FT8_CHAR_TABLE_FULL) + c14

    return c14.strip()
