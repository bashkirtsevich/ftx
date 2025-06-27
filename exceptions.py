class FTXException(Exception):
    pass


class FTXErrorCallSign1(FTXException):
    pass


class FTXErrorCallSign2(FTXException):
    pass


class FTXErrorTooLong(FTXException):
    pass


class FTXErrorInvalidChar(FTXException):
    pass


class FTXErrorSuffix(FTXException):
    pass


class FTXErrorGrid(FTXException):
    pass


class FTXNotImplemented(FTXException):
    pass


class FTXInvalidCallsign(FTXException):
    pass


class FTXInvalidReport(FTXException):
    pass


class FTXPack28Error(FTXException):
    pass


class FTXErrorMsgType(FTXException):
    pass
