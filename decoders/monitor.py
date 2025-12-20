import numpy.typing as npt

import typing
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass

DecodeStatus = namedtuple("DecodeStatus", ["ldpc_errors", "crc_extracted"])


@dataclass
class LogItem:
    snr: float
    dT: float
    dF: float
    payload: typing.ByteString
    crc: int


class AbstractMonitor(ABC):
    @staticmethod
    def pack_bits(bit_array: typing.ByteString, num_bits: int) -> typing.ByteString:
        # Packs a string of bits each represented as a zero/non-zero byte in plain[],
        # as a string of packed bits starting from the MSB of the first byte of packed[]
        num_bytes = (num_bits + 7) // 8
        packed = bytearray(b"\x00" * num_bytes)

        mask = 0x80
        byte_idx = 0
        for i in range(num_bits):
            if bit_array[i]:
                packed[byte_idx] |= mask

            mask >>= 1
            if not mask:
                mask = 0x80
                byte_idx += 1

        return packed

    @abstractmethod
    def monitor_process(self, frame: npt.NDArray):
        ...

    @abstractmethod
    def decode(self, **kwargs) -> typing.Generator[LogItem, None, None]:
        ...
