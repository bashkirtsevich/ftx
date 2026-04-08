import typing
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .encoder import ENCODE
from .node import Node
from .poly import LL_POLY1, LL_POLY2


@dataclass
class NodeJelinek(Node):
    depth: int  # depth of this node
    jpointer: int


def jelinek(
        symbols: npt.NDArray[np.uint8],  # Raw deinterleaved input symbols
        bits: int,  # Number of output bits
        # stack: typing.List[Node],
        stack_size: int,
        metric: npt.NDArray[np.int64],  # Metric table, [sent sym][rx symbol]
        max_iter: int,  # Decoding timeout in cycles per bit
        poly1: int = LL_POLY1,
        poly2: int = LL_POLY2,
) -> typing.Optional[typing.Tuple[int, int, npt.NDArray[np.uint8]]]:
    # Compute branch metrics for each symbol pair
    # The sequential decoding algorithm only uses the metrics, not the
    # symbol values.

    metrics = np.zeros((81, 4), dtype=np.int64)
    for i in range(bits):
        s0 = symbols[i * 2]
        s1 = symbols[i * 2 + 1]
        metrics[i, 0] = metric[0, s0] + metric[0, s1]
        metrics[i, 1] = metric[0, s0] + metric[1, s1]
        metrics[i, 2] = metric[1, s0] + metric[0, s1]
        metrics[i, 3] = metric[1, s0] + metric[1, s1]

    # zero the stack
    stack = [NodeJelinek(0, 0, 0, 0) for _ in range(stack_size)]
    # initialize the loop variables
    ntail = 31
    encstate = 0
    nbuckets = 1000
    low_bucket = nbuckets - 1  # will be set on first run-through
    high_bucket = 0
    # unsigned int *buckets, bucket;
    # buckets=malloc(nbuckets*sizeof(unsigned int));
    buckets = np.zeros(nbuckets, dtype=np.int64)
    # memset(buckets,0,nbuckets*sizeof(unsigned int));
    ptr = 1
    stackptr = 1  # pointer values of 0 are reserved (they mean that a bucket is empty)
    depth = 0
    nbits_minus_ntail = bits - ntail
    stacksize_minus_1 = len(stack) - 1
    # long int totmet0, totmet1,
    gamma = 0
    #
    ncycles = max_iter * bits
    # ********************* Start the stack decoder *****************
    for i in range(1, ncycles + 1):
        # printf("***stackptr=%ld, depth=%d, gamma=%d, encstate=%lx, bucket %d, low_bucket %d, high_bucket %d\n",
        #        stackptr, depth, gamma, encstate, bucket, low_bucket, high_bucket);
        # no need to store more than 7 bytes (56 bits) for encoder state because
        # only 50 bits are not 0's.
        if depth < 56:
            encstate <<= 1
            lsym = ENCODE(encstate, poly1, poly2)  # get channel symbols associated with the 0 branch
        else:
            lsym = ENCODE(encstate << (depth - 55), poly1, poly2)

        # lsym are the 0-branch channel symbols and 3^lsym are the 1-branch
        # channel symbols (due to a special property of our generator polynomials)
        totmet0 = gamma + metrics[depth][lsym]  # total metric for 0-branch daughter node
        totmet1 = gamma + metrics[depth][3 ^ lsym]  # total metric for 1-branch daughter node
        depth += 1  # the depth of the daughter nodes

        bucket = (totmet0 >> 5) + 200  # fast, but not particularly safe - totmet can be negative
        high_bucket = max(high_bucket, bucket)
        low_bucket = min(low_bucket, bucket)

        # place the 0 node on the stack, overwriting the parent (current) node
        stack[ptr].encstate = encstate
        stack[ptr].gamma = totmet0
        stack[ptr].depth = depth
        stack[ptr].jpointer = buckets[bucket]
        buckets[bucket] = ptr

        # if in the tail, only need to evaluate the "0" branch.
        # Otherwise, enter this "if" and place the 1 node on the stack,
        if depth <= nbits_minus_ntail:
            if stackptr < stacksize_minus_1:
                stackptr += 1
                ptr = stackptr
            else:  # stack full
                while buckets[low_bucket] == 0:  # write latest to where the top of the lowest bucket points
                    low_bucket += 1

                ptr = buckets[low_bucket]
                buckets[low_bucket] = stack[ptr].jpointer  # make bucket point to next older entry

            bucket = (totmet1 >> 5) + 200  # this may not be safe on all compilers
            if bucket > high_bucket:
                high_bucket = bucket
            if bucket < low_bucket:
                low_bucket = bucket

            stack[ptr].encstate = encstate + 1
            stack[ptr].gamma = totmet1
            stack[ptr].depth = depth
            stack[ptr].jpointer = buckets[bucket]
            buckets[bucket] = ptr

        # pick off the latest entry from the high bucket
        while buckets[high_bucket] == 0:
            high_bucket -= 1

        ptr = buckets[high_bucket]
        buckets[high_bucket] = stack[ptr].jpointer
        depth = stack[ptr].depth
        gamma = stack[ptr].gamma
        encstate = stack[ptr].encstate

        # we are done if the top entry on the stack is at depth nbits
        if depth == bits:
            break

    cycles = i + 1
    metric = gamma  # Return final path metric

    #    printf("cycles %d stackptr=%d, depth=%d, gamma=%d, encstate=%lx\n",
    #           *cycles, stackptr, depth, *metric, encstate);

    data = np.zeros(7, dtype=np.int64)
    for i in range(7):
        data[i] = (encstate >> (48 - i * 8)) & 0xff

    # for i in range(7, 11):
    #     data[i] = 0

    if cycles / bits >= max_iter:  # timed out
        return None

    return metric, cycles, data
