import math
import typing

from consts import kFT4_XOR_sequence, FT4_ND, FTX_LDPC_N, FT8_ND, kFT4_Gray_map, kFT8_Gray_map, FTX_LDPC_K, \
    kFT8_Costas_pattern, FT8_LENGTH_SYNC, FT8_NUM_SYNC, FT8_SYNC_OFFSET, FT4_NUM_SYNC, kFT4_Costas_pattern, \
    FT4_LENGTH_SYNC, FT4_SYNC_OFFSET, FTX_PROTOCOL_FT4
from crc import ftx_extract_crc, ftx_compute_crc
from ldpc import bp_decode
from message import ftx_message_decode

kMin_score = 10  # Minimum sync score threshold for candidates
kMax_candidates = 140
kLDPC_iterations = 25

kMax_decoded_messages = 50

kFreq_osr = 2  # Frequency oversampling rate (bin subdivision)
kTime_osr = 2  # Time oversampling rate (symbol subdivision)


# Input structure to ftx_find_sync() function. This structure describes stored waterfall data over the whole message slot.
# Fields time_osr and freq_osr specify additional oversampling rate for time and frequency resolution.
# If time_osr=1, FFT magnitude data is collected once for every symbol transmitted, i.e. every 1/6.25 = 0.16 seconds.
# Values time_osr > 1 mean each symbol is further subdivided in time.
# If freq_osr=1, each bin in the FFT magnitude data corresponds to 6.25 Hz, which is the tone spacing.
# Values freq_osr > 1 mean the tone spacing is further subdivided by FFT analysis.
class ftx_waterfall_t:
    def __init__(self):
        self.max_blocks: int = 0  # < number of blocks (symbols) allocated in the mag array
        self.num_blocks: int = 0  # < number of blocks (symbols) stored in the mag array
        self.num_bins: int = 0  # < number of FFT bins in terms of 6.25 Hz
        self.time_osr: int = 0  # < number of time subdivisions
        self.freq_osr: int = 0  # < number of frequency subdivisions
        self.mag: typing.List[int] = []  # FFT magnitudes stored as uint8_t[blocks][time_osr][freq_osr][num_bins]
        self.block_stride: int = 0  # < Helper value = time_osr * freq_osr * num_bins
        self.protocol: int = 0  # < Indicate if using FT4 or FT8


# Output structure of ftx_find_sync() and input structure of ftx_decode().
# Holds the position of potential start of a message in time and frequency.
class ftx_candidate_t:
    def __init__(self, time_offset, freq_offset, time_sub, freq_sub):
        self.score: int = 0  # < Candidate score (non-negative number; higher score means higher likelihood)
        self.time_offset: int = time_offset  # < Index of the time block
        self.freq_offset: int = freq_offset  # < Index of the frequency bin
        self.time_sub: int = time_sub  # < Index of the time subdivision used
        self.freq_sub: int = freq_sub  # < Index of the frequency subdivision used

    def __repr__(self):
        return str(self)

    def __str__(self):
        return (f"[score: {self.score}; "
                f"time_offset: {self.time_offset}; "
                f"freq_offset: {self.freq_offset}; "
                f"time_sub: {self.time_sub}; "
                f"freq_sub: {self.freq_sub}]")


# Structure that contains the status of various steps during decoding of a message
class ftx_decode_status_t:
    def __init__(self):
        self.freq: float = 0.0
        self.time: float = 0.0
        self.ldpc_errors: int = 0  # < Number of LDPC errors during decoding
        self.crc_extracted: int = 0  # < CRC value recovered from the message
        self.crc_calculated: int = 0  # < CRC value calculated over the payload
        # int unpack_status;     #< Return value of the unpack routine


# FT4/FT8 monitor object that manages DSP processing of incoming audio data
# and prepares a waterfall object
class monitor_t:
    def __init__(self):
        self.symbol_period: float = 0  # < FT4/FT8 symbol period in seconds
        self.min_bin: int = 0  # < First FFT bin in the frequency range (begin)
        self.max_bin: int = 0  # < First FFT bin outside the frequency range (end)
        self.block_size: int = 0  # < Number of samples per symbol (block)
        self.subblock_size: int = 0  # < Analysis shift size (number of samples)
        self.nfft: int = 0  # < FFT size
        self.fft_norm: float = 0  # < FFT normalization factor
        self.window: typing.List[float] = []  # < Window function for STFT analysis (nfft samples)
        self.last_frame: typing.List[float] = []  # < Current STFT analysis frame (nfft samples)
        self.wf: ftx_waterfall_t = None  # < Waterfall object
        self.max_mag: float = 0  # < Maximum detected magnitude (debug stats)

        # KISS FFT housekeeping variables
        # void* fft_work;        #< Work area required by Kiss FFT
        # kiss_fftr_cfg fft_cfg; #< Kiss FFT housekeeping object
    # #ifdef WATERFALL_USE_PHASE
    #     int nifft;             #< iFFT size
    #     void* ifft_work;       #< Work area required by inverse Kiss FFT
    #     kiss_fft_cfg ifft_cfg; #< Inverse Kiss FFT housekeeping object
    # #endif


# Packs a string of bits each represented as a zero/non-zero byte in plain[],
# as a string of packed bits starting from the MSB of the first byte of packed[]
def pack_bits(bit_array: bytes, num_bits: int) -> bytes:  # (const uint8_t bit_array[], int num_bits, uint8_t packed[])
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


def ft8_sync_score(wf: ftx_waterfall_t, candidate: ftx_candidate_t) -> int:
    score = 0
    num_average = 0

    # Get the pointer to symbol 0 of the candidate
    mag_cand = get_cand_mag_idx(wf, candidate)

    # Compute average score over sync symbols (m+k = 0-7, 36-43, 72-79)
    for m in range(FT8_NUM_SYNC):
        for k in range(FT8_LENGTH_SYNC):
            block = (FT8_SYNC_OFFSET * m) + k  # relative to the message
            block_abs = candidate.time_offset + block  # relative to the captured signal

            if block_abs < 0:  # Check for time boundaries
                continue

            if block_abs >= wf.num_blocks:
                break

            # Get the pointer to symbol 'block' of the candidate
            p8 = mag_cand + (block * wf.block_stride)

            # Check only the neighbors of the expected symbol frequency- and time-wise
            sm = kFT8_Costas_pattern[k]  # Index of the expected bin
            if sm > 0:  # look at one frequency bin lower
                score += wf.mag[p8 + sm] - wf.mag[p8 + sm - 1]
                num_average += 1
            if sm < 7:  # look at one frequency bin higher
                score += wf.mag[p8 + sm] - wf.mag[p8 + sm + 1]
                num_average += 1
            if k > 0 and block_abs > 0:  # look one symbol back in time
                score += wf.mag[p8 + sm] - wf.mag[p8 + sm - wf.block_stride]
                num_average += 1
            if ((k + 1) < FT8_LENGTH_SYNC) and ((block_abs + 1) < wf.num_blocks):  # look one symbol forward in time
                score += wf.mag[p8 + sm] - wf.mag[p8 + sm + wf.block_stride]
                num_average += 1

    if num_average > 0:
        score = int(score / num_average)

    # if score != 0:
    #     print(f"ft8_sync_score score={score}")
    return score


def ft4_sync_score(wf: ftx_waterfall_t, candidate: ftx_candidate_t) -> int:
    score = 0
    num_average = 0

    # Get the pointer to symbol 0 of the candidate
    mag_cand = get_cand_mag_idx(wf, candidate)

    # Compute average score over sync symbols (block = 1-4, 34-37, 67-70, 100-103)
    for m in range(FT4_NUM_SYNC):
        for k in range(FT4_LENGTH_SYNC):
            block = 1 + (FT4_SYNC_OFFSET * m) + k
            block_abs = candidate.time_offset + block
            # Check for time boundaries
            if block_abs < 0:
                continue
            if block_abs >= wf.num_blocks:
                break

            # Get the pointer to symbol 'block' of the candidate
            p4 = mag_cand + (block * wf.block_stride)

            sm = kFT4_Costas_pattern[m][k]  # Index of the expected bin

            # score += (4 * p4[sm]) - p4[0] - p4[1] - p4[2] - p4[3];
            # num_average += 4;

            # Check only the neighbors of the expected symbol frequency- and time-wise
            if sm > 0:
                # look at one frequency bin lower
                # score += WF_ELEM_MAG_INT(p4[sm]) - WF_ELEM_MAG_INT(p4[sm - 1]);
                score += int(p4[sm]) - int(p4[sm - 1])
                num_average += 1
            if sm < 3:
                # look at one frequency bin higher
                # score += WF_ELEM_MAG_INT(p4[sm]) - WF_ELEM_MAG_INT(p4[sm + 1]);
                score += int(p4[sm]) - int(p4[sm + 1])
                num_average += 1
            if k > 0 and block_abs > 0:
                # look one symbol back in time
                # score += WF_ELEM_MAG_INT(p4[sm]) - WF_ELEM_MAG_INT(p4[sm - wf->block_stride]);
                score += int(p4[sm]) - int(p4[sm - wf.block_stride])
                num_average += 1
            if (k + 1) < FT4_LENGTH_SYNC and (block_abs + 1) < wf.num_blocks:
                # look one symbol forward in time
                # score += WF_ELEM_MAG_INT(p4[sm]) - WF_ELEM_MAG_INT(p4[sm + wf->block_stride]);
                score += int(p4[sm]) - int(p4[sm + wf.block_stride])
                num_average += 1

    if num_average > 0:
        score = int(score / num_average)

    return score


# (const ftx_waterfall_t* wf, int num_candidates, ftx_candidate_t heap[], int min_score)
def ftx_find_candidates(wf: ftx_waterfall_t, num_candidates: int, min_score: int) -> list:
    if wf.protocol == FTX_PROTOCOL_FT4:
        sync_fun = ft4_sync_score
    else:
        sync_fun = ft8_sync_score

    num_tones = 4 if wf.protocol == FTX_PROTOCOL_FT4 else 8

    heap = []

    # Here we allow time offsets that exceed signal boundaries, as long as we still have all data bits.
    # I.e. we can afford to skip the first 7 or the last 7 Costas symbols, as long as we track how many
    # sync symbols we included in the score, so the score is averaged.
    for time_sub in range(wf.time_osr):
        for freq_sub in range(wf.freq_osr):
            for time_offset in range(-10, 20):
                for freq_offset in range(
                        wf.num_bins - num_tones):  # (candidate.freq_offset + num_tones - 1) < wf->num_bin
                    candidate = ftx_candidate_t(time_sub=time_sub, freq_sub=freq_sub, time_offset=time_offset,
                                                freq_offset=freq_offset)

                    if (score := sync_fun(wf, candidate)) >= min_score:
                        candidate.score = score
                        heap.insert(0, candidate)

    heap.sort(key=lambda x: x.score, reverse=True)
    return heap[:num_candidates]


def get_cand_mag_idx(wf: ftx_waterfall_t, candidate: ftx_candidate_t) -> int:
    offset = candidate.time_offset
    offset = (offset * wf.time_osr) + candidate.time_sub
    offset = (offset * wf.freq_osr) + candidate.freq_sub
    offset = (offset * wf.num_bins) + candidate.freq_offset

    return offset


def ft4_extract_likelihood(wf: ftx_waterfall_t, cand: ftx_candidate_t) -> typing.List[float]:
    log174 = [0.0] * FTX_LDPC_N

    mag = get_cand_mag_idx(wf, cand)  # Pointer to 4 magnitude bins of the first symbol

    # Go over FSK tones and skip Costas sync symbols
    for k in range(FT4_ND):
        # Skip either 5, 9 or 13 sync symbols
        # TODO: replace magic numbers with constants
        sym_idx = k + (5 if k < 29 else 9 if k < 58 else 13)
        bit_idx = 2 * k

        # Check for time boundaries
        block = cand.time_offset + sym_idx
        if block < 0 or block >= wf.num_blocks:
            log174[bit_idx + 0] = 0
            log174[bit_idx + 1] = 0
        else:
            logl_0, logl_1 = ft4_extract_symbol(wf, mag + sym_idx * wf.block_stride)
            log174[bit_idx + 0] = logl_0
            log174[bit_idx + 1] = logl_1

    return log174


def ft8_extract_likelihood(wf: ftx_waterfall_t, cand: ftx_candidate_t) -> typing.List[float]:
    log174 = [0.0] * FTX_LDPC_N
    mag = get_cand_mag_idx(wf, cand)  # Pointer to 8 magnitude bins of the first symbol

    # Go over FSK tones and skip Costas sync symbols
    for k in range(FT8_ND):
        # Skip either 7 or 14 sync symbols
        # TODO: replace magic numbers with constants
        sym_idx = k + (7 if k < 29 else 14)
        bit_idx = 3 * k

        # Check for time boundaries
        block = cand.time_offset + sym_idx

        if block < 0 or block >= wf.num_blocks:
            log174[bit_idx + 0] = 0
            log174[bit_idx + 1] = 0
            log174[bit_idx + 2] = 0
        else:
            logl_0, logl_1, logl_2 = ft8_extract_symbol(wf, mag + sym_idx * wf.block_stride)
            log174[bit_idx + 0] = logl_0
            log174[bit_idx + 1] = logl_1
            log174[bit_idx + 2] = logl_2

    return log174


# Compute unnormalized log likelihood log(p(1) / p(0)) of 2 message bits (1 FSK symbol)
def ft4_extract_symbol(wf: ftx_waterfall_t, mag_idx: int) -> typing.Tuple[float, float]:
    # Cleaned up code for the simple case of n_syms==1
    s2 = [wf.mag[mag_idx + kFT4_Gray_map[j]] for j in range(4)]

    logl_0 = max(s2[2], s2[3]) - max(s2[0], s2[1])
    logl_1 = max(s2[1], s2[3]) - max(s2[0], s2[2])

    return logl_0, logl_1


# Compute unnormalized log likelihood log(p(1) / p(0)) of 3 message bits (1 FSK symbol)
def ft8_extract_symbol(wf: ftx_waterfall_t, mag_idx: int) -> typing.Tuple[float, float, float]:
    # Cleaned up code for the simple case of n_syms==1
    s2 = [wf.mag[mag_idx + kFT8_Gray_map[j]] for j in range(8)]

    logl_0 = max(s2[4], s2[5], s2[6], s2[7]) - max(s2[0], s2[1], s2[2], s2[3])
    logl_1 = max(s2[2], s2[3], s2[6], s2[7]) - max(s2[0], s2[1], s2[4], s2[5])
    logl_2 = max(s2[1], s2[3], s2[5], s2[7]) - max(s2[0], s2[2], s2[4], s2[6])

    return logl_0, logl_1, logl_2


def ftx_normalize_logl(log174: typing.List[float]) -> typing.List[float]:
    # FIXME: Optimize
    # Compute the variance of log174
    sum = 0
    sum2 = 0
    for it in log174:
        sum += it
        sum2 += it ** 2

    inv_n = 1.0 / FTX_LDPC_N
    variance = (sum2 - (sum * sum * inv_n)) * inv_n

    # Normalize log174 distribution and scale it with experimentally found coefficient
    norm_factor = math.sqrt(24.0 / variance)
    return [it * norm_factor for it in log174]


def ftx_decode_candidate(
        wf: ftx_waterfall_t, cand: ftx_candidate_t,
        max_iterations: int) -> typing.Optional[typing.Tuple[ftx_decode_status_t, typing.Optional[bytes]]]:
    if wf.protocol == FTX_PROTOCOL_FT4:
        log174 = ft4_extract_likelihood(wf, cand)
    else:
        log174 = ft8_extract_likelihood(wf, cand)

    log174 = ftx_normalize_logl(log174)

    status = ftx_decode_status_t()
    status.ldpc_errors, plain174 = bp_decode(log174, max_iterations)

    if status.ldpc_errors > 0:
        return status, None

    # Extract payload + CRC (first FTX_LDPC_K bits) packed into a byte array
    a91 = pack_bits(plain174, FTX_LDPC_K)

    # Extract CRC and check it
    status.crc_extracted = ftx_extract_crc(a91)
    # [1]: 'The CRC is calculated on the source-encoded message, zero-extended from 77 to 82 bits.'
    a91[9] &= 0xF8
    a91[10] &= 0x00
    status.crc_calculated = ftx_compute_crc(a91, 96 - 14)

    if status.crc_extracted != status.crc_calculated:
        return status, None

    # Reuse CRC value as a hash for the message (TODO: 14 bits only, should perhaps use full 16 or 32 bits?)
    # message.hash = status.crc_calculated

    if wf.protocol == FTX_PROTOCOL_FT4:
        # '[..] for FT4 only, in order to avoid transmitting a long string of zeros when sending CQ messages,
        # the assembled 77-bit message is bitwise exclusive-OR’ed with [a] pseudorandom sequence before computing the CRC and FEC parity bits'
        payload = bytearray(b"\x00" * 10)
        for i in range(10):
            payload[i] = a91[i] ^ kFT4_XOR_sequence[i]
    else:
        payload = a91

    return status, payload


def decode(mon: monitor_t, tm_slot_start):  # (const monitor_t* mon, struct tm* tm_slot_start)
    wf = mon.wf
    # Find top candidates by Costas sync score and localize them in time and frequency
    # candidate_list[kMax_candidates]
    candidate_list = ftx_find_candidates(wf, kMax_candidates, kMin_score)

    # Hash table for decoded messages (to check for duplicates)
    num_decoded = 0
    # decoded[kMax_decoded_messages]
    # decoded_hashtable[kMax_decoded_messages]

    # Initialize hash table pointers
    # for i in range(kMax_decoded_messages):
    #     decoded_hashtable[i] = NULL

    # Go over candidates and attempt to decode messages
    # for idx in range(num_candidates):
    #     cand = &candidate_list[idx]
    for cand in candidate_list:
        freq_hz = (mon.min_bin + cand.freq_offset + cand.freq_sub / wf.freq_osr) / mon.symbol_period
        time_sec = (cand.time_offset + cand.time_sub / wf.time_osr) * mon.symbol_period

        # ftx_message_t message;
        # ftx_decode_status_t status;
        status, message = ftx_decode_candidate(wf, cand, kLDPC_iterations)
        if not message:
            # if status.ldpc_errors > 0:
            #     print(f"LDPC decode: {status.ldpc_errors} errors")
            # elif status.crc_calculated != status.crc_extracted:
            #     print("CRC mismatch!")
            continue

        # LOG(LOG_DEBUG, "Checking hash table for %4.1fs / %4.1fHz [%d]...\n", time_sec, freq_hz, cand->score);
        # print(f"Checking hash table for {time_sec} / {freq_hz}Hz [{cand.score}]")
        # idx_hash = message.hash % kMax_decoded_messages
        # found_empty_slot = False
        # found_duplicate = False

        #     while True:
        #         if not decoded_hashtable[idx_hash]:
        #             # LOG(LOG_DEBUG, "Found an empty slot\n")
        #             found_empty_slot = True
        #         elif ((decoded_hashtable[idx_hash].hash == message.hash) and (0 == memcmp(decoded_hashtable[idx_hash]->payload, message.payload, sizeof(message.payload)))):
        #             # LOG(LOG_DEBUG, "Found a duplicate!\n")
        #             found_duplicate = True
        #         else:
        #             # LOG(LOG_DEBUG, "Hash table clash!\n");
        #             # Move on to check the next entry in hash table
        #             idx_hash = (idx_hash + 1) % kMax_decoded_messages
        #
        #         if found_empty_slot or found_duplicate:
        #             break
        #
        #
        if True:  # found_empty_slot:
            #         # Fill the empty hashtable slot
            #         memcpy(&decoded[idx_hash], &message, sizeof(message))
            #         decoded_hashtable[idx_hash] = decoded[idx_hash]
            #         num_decoded += 1
            #
            #         # text[FTX_MAX_MESSAGE_LENGTH]
            #         unpack_status = ftx_message_decode(&message, &hash_if, text)
            #         if (unpack_status != FTX_MESSAGE_RC_OK)
            #             snprintf(text, sizeof(text), "Error [%d] while unpacking!", (int)unpack_status);
            #
            #         # Fake WSJT-X-like output for now
            snr = cand.score * 0.5  # TODO: compute better approximation of SNR
            # print("%02d%02d%02d %+05.1f %+4.2f %4.0f ~  %s\n",
            #     tm_slot_start->tm_hour, tm_slot_start->tm_min, tm_slot_start->tm_sec,
            #     snr, time_sec, freq_hz, text)
            call_to_rx, call_de_rx, extra_rx = ftx_message_decode(message)
            print("DECODE:", snr, time_sec, freq_hz, call_to_rx, call_de_rx, extra_rx)

    #
    # # LOG(LOG_INFO, "Decoded %d messages, callsign hashtable size %d\n", num_decoded, callsign_hashtable_size);
    # # hashtable_cleanup(10)
