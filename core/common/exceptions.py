class SignalAnalyzerError(Exception):
    """Base exception for the signal analyzer."""
    pass

class UnsupportedFormatError(SignalAnalyzerError):
    """Raised when a file format is not supported."""
    pass

class InvalidMetadataError(SignalAnalyzerError):
    """Raised when metadata is invalid or inconsistent."""
    pass

class InvalidIQConfigurationError(SignalAnalyzerError):
    """Raised when IQ loading configuration is invalid or incomplete."""
    pass

class SignalLoadError(SignalAnalyzerError):
    """Raised when signal loading fails for any reason."""
    pass

class ValidationError(SignalAnalyzerError):
    """Raised when validation fails."""
    pass
