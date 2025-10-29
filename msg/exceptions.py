class MSGException(Exception):
    pass


class MSGErrorCallSignTo(MSGException):
    pass


class MSGErrorCallSignDe(MSGException):
    pass


class MSGErrorTooLong(MSGException):
    pass


class MSGErrorInvalidChar(MSGException):
    pass


class MSGErrorSuffix(MSGException):
    pass


class MSGErrorGrid(MSGException):
    pass


class MSGNotImplemented(MSGException):
    pass


class MSGInvalidCallsign(MSGException):
    pass


class MSGInvalidReport(MSGException):
    pass


class MSGPackCallsignError(MSGException):
    pass


class MSGErrorMsgType(MSGException):
    pass
