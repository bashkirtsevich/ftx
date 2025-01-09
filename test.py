from message import ftx_message_encode, ftx_message_decode


def test_std_msg(call_to_tx, call_de_tx, extra_tx):
    payload = ftx_message_encode(call_to_tx, call_de_tx, extra_tx)
    call_to_rx, call_de_rx, extra_rx = ftx_message_decode(payload)

    assert call_to_tx == call_to_rx
    assert call_de_tx == call_de_rx
    assert extra_tx == extra_rx


def main():
    callsigns = ["YL3JG", "W1A", "W1A/R", "W5AB", "W8ABC", "DE6ABC", "DE6ABC/R", "DE7AB", "DE9A", "3DA0X", "3DA0XYZ",
                 "3DA0XYZ/R", "3XZ0AB", "3XZ0A"]
    tokens = ["CQ", "QRZ"]
    grids = ["KO26", "RR99", "AA00", "RR09", "AA01", "RRR", "RR73", "73", "R+10", "R+05", "R-12", "R-02", "+10", "+05",
             "-02", "-02", ""]

    for grid in grids:
        for callsign in callsigns:
            for callsign2 in callsigns:
                test_std_msg(callsign, callsign2, grid)
        for token in tokens:
            for callsign2 in callsigns:
                test_std_msg(token, callsign2, grid)


if __name__ == '__main__':
    main()
