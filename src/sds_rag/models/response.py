"""
Standardized API response wrapper for consistent response format.
"""

from typing import Any, Optional, Dict, List
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from ..utils.datetime_utils import utc_now_iso


class ResponseStatus(str, Enum):
    """Response status enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


class ApiResponse(BaseModel):
    """Standardized API response wrapper."""
    
    status: ResponseStatus = Field(..., description="Response status")
    message: str = Field(..., description="Human-readable message")
    data: Optional[Any] = Field(None, description="Response data payload")
    errors: Optional[List[str]] = Field(None, description="List of error messages")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    timestamp: str = Field(default_factory=utc_now_iso, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request tracking ID")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SuccessResponse(ApiResponse):
    """Success response wrapper."""
    
    def __init__(
        self, 
        data: Any = None, 
        message: str = "Request completed successfully",
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            status=ResponseStatus.SUCCESS,
            message=message,
            data=data,
            metadata=metadata,
            request_id=request_id,
            **kwargs
        )


class ErrorResponse(ApiResponse):
    """Error response wrapper."""
    
    def __init__(
        self, 
        message: str = "An error occurred",
        errors: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            status=ResponseStatus.ERROR,
            message=message,
            errors=errors or [],
            metadata=metadata,
            request_id=request_id,
            **kwargs
        )


class WarningResponse(ApiResponse):
    """Warning response wrapper."""
    
    def __init__(
        self, 
        data: Any = None,
        message: str = "Request completed with warnings",
        errors: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            status=ResponseStatus.WARNING,
            message=message,
            data=data,
            errors=errors or [],
            metadata=metadata,
            request_id=request_id,
            **kwargs
        )


def success_response(
    data: Any = None,
    message: str = "Request completed successfully",
    metadata: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> ApiResponse:
    """Create a success response."""
    return SuccessResponse(
        data=data,
        message=message,
        metadata=metadata,
        request_id=request_id
    )


def error_response(
    message: str = "An error occurred",
    errors: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> ApiResponse:
    """Create an error response."""
    return ErrorResponse(
        message=message,
        errors=errors,
        metadata=metadata,
        request_id=request_id
    )


def warning_response(
    data: Any = None,
    message: str = "Request completed with warnings",
    errors: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> ApiResponse:
    """Create a warning response."""
    return WarningResponse(
        data=data,
        message=message,
        errors=errors,
        metadata=metadata,
        request_id=request_id
    )