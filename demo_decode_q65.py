import numpy as np
import json

from decoders.qra import Q65


def main():
    with open("examples/data3.json") as f:
        data = json.load(f)

    iwave = np.array(data["iwave"], dtype=np.float64)

    q65 = Q65()
    q65.q65_dec0(
        iavg=0,
        iwave=iwave,
        nfqso=1000,
        lclearave=False,
        emedelay=False,
        nQSOp=0,
        cont_id=0,
        cont_type=0,
        stageno=0,
        fsdec=False,
    )


if __name__ == '__main__':
    # python -m cProfile -s time demo_decode_q65.py
    main()
