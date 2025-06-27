class FTXException(Exception):
    pass


class FTXErrorCallSignTo(FTXException):
    pass


class FTXErrorCallSignDe(FTXException):
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


class FTXPackCallsignError(FTXException):
    pass


class FTXErrorMsgType(FTXException):
    pass
