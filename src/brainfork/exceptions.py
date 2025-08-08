"""
Custom exceptions for Brainfork
"""


class BrainforkError(Exception):
    """Base exception for Brainfork"""
    pass


class ModelNotFoundError(BrainforkError):
    """Raised when a requested model is not found in configuration"""
    pass


class AuthenticationError(BrainforkError):
    """Raised when authentication fails"""
    pass


class RoutingError(BrainforkError):
    """Raised when routing decision fails"""
    pass


class ConfigurationError(BrainforkError):
    """Raised when configuration is invalid"""
    pass
