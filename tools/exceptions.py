"""Exceptions raised by the tool layer."""


class ToolError(Exception):
    """Base class for expected tool execution errors."""


class ToolInputError(ToolError):
    """Raised when a tool receives invalid input."""


class ToolExecutionError(ToolError):
    """Raised when a tool cannot complete a valid request."""

