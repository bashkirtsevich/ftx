class QRAException(Exception):
    ...


class InvalidQRAType(QRAException):
    ...


class InvalidFadingModel(QRAException):
    ...


class CRCMismatch(QRAException):
    ...


class MExceeded(QRAException):
    ...


class DecodeFailed(QRAException):
    ...
