import enum


class Type_Error(enum.Enum):
    pars_error = 0


class Error(object):
    def __init__(self, type_error: Type_Error, message=""):
        self.type = type_error
        self.message = message
