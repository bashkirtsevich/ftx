class TextException(Exception):
    ...


class CharTableMissmatch(TextException):
    ...


class MSGException(Exception):
    ...


class MSGErrorCallSignTo(MSGException):
    ...


class MSGErrorCallSignDe(MSGException):
    ...


class MSGErrorTooLong(MSGException):
    ...


class MSGErrorInvalidChar(MSGException):
    ...


class MSGErrorSuffix(MSGException):
    ...


class MSGErrorGrid(MSGException):
    ...


class MSGNotImplemented(MSGException):
    ...


class MSGInvalidCallsign(MSGException):
    ...


class MSGInvalidReport(MSGException):
    ...


class MSGPackCallsignError(MSGException):
    ...


class MSGErrorMsgType(MSGException):
    ...


class MSGValidateError(MSGException):
    ...
