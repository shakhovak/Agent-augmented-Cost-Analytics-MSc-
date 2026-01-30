"""Custom exceptions for the cost agent."""


from __future__ import annotations


class CostAgentError(Exception):
    """Base exception for the project."""


class ConfigError(CostAgentError):
    """Raised when configuration files are missing/invalid."""


class DataSourceError(CostAgentError):
    """Raised when the data source cannot be read or validated."""


class UnsupportedQuery(CostAgentError):
    """Raised when a query requests unsupported fields/operations."""


class RowLimitExceeded(CostAgentError):
    """Raised when a query would exceed row safety limits."""


class TimeWindowExceeded(CostAgentError):
    """Raised when a query requests too large a date range."""


class ValidationError(CostAgentError):
    """Raised for invalid input values (e.g., malformed dates)."""
