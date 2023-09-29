"""
This module will host all the exceptions which are raised by ood_enabler components
"""


class OODEnableException(Exception):
    pass


class UnknownMLBackend(OODEnableException):
    pass


class UnknownMLArch(OODEnableException):
    pass


class UnknownInferenceService(OODEnableException):
    pass


class UnknownExtension(OODEnableException):
    pass
