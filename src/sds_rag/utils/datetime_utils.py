"""
Date and time utilities for SDS-RAG system.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Union
import time
import logging

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    """
    Get current UTC datetime.
    
    Returns:
        datetime: Current UTC datetime
    """
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """
    Get current UTC datetime as ISO string.
    
    Returns:
        str: Current UTC datetime in ISO format
    """
    return utc_now().isoformat()


def format_timestamp(dt: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    """
    Format a datetime as a string.
    
    Args:
        dt: Datetime to format (uses current UTC if None)
        format_str: Format string
        
    Returns:
        str: Formatted datetime string
    """
    if dt is None:
        dt = utc_now()
    
    return dt.strftime(format_str)


def parse_iso_timestamp(timestamp_str: str) -> datetime:
    """
    Parse an ISO timestamp string to datetime.
    
    Args:
        timestamp_str: ISO timestamp string
        
    Returns:
        datetime: Parsed datetime object
        
    Raises:
        ValueError: If timestamp string is invalid
    """
    try:
        # Handle different ISO formats
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'
        
        return datetime.fromisoformat(timestamp_str)
    except ValueError as e:
        logger.error(f"Failed to parse timestamp '{timestamp_str}': {e}")
        raise


def seconds_to_human(seconds: float) -> str:
    """
    Convert seconds to human-readable duration.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Human-readable duration (e.g., "2h 30m 45s")
    """
    if seconds < 0:
        return "0s"
    
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    
    if minutes < 60:
        if remaining_seconds > 0:
            return f"{int(minutes)}m {remaining_seconds:.0f}s"
        return f"{int(minutes)}m"
    
    hours = minutes // 60
    remaining_minutes = minutes % 60
    
    if hours < 24:
        parts = [f"{int(hours)}h"]
        if remaining_minutes > 0:
            parts.append(f"{int(remaining_minutes)}m")
        if remaining_seconds > 0 and hours < 2:  # Only show seconds for short durations
            parts.append(f"{remaining_seconds:.0f}s")
        return " ".join(parts)
    
    days = hours // 24
    remaining_hours = hours % 24
    
    parts = [f"{int(days)}d"]
    if remaining_hours > 0:
        parts.append(f"{int(remaining_hours)}h")
    if remaining_minutes > 0 and days < 2:
        parts.append(f"{int(remaining_minutes)}m")
    
    return " ".join(parts)


def time_ago(dt: datetime, reference: Optional[datetime] = None) -> str:
    """
    Get human-readable time difference (e.g., "2 hours ago").
    
    Args:
        dt: Past datetime
        reference: Reference datetime (uses current UTC if None)
        
    Returns:
        str: Human-readable time ago string
    """
    if reference is None:
        reference = utc_now()
    
    # Ensure both datetimes are timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    
    diff = reference - dt
    
    if diff.total_seconds() < 0:
        return "in the future"
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return "just now"
    
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)} minute{'s' if int(minutes) != 1 else ''} ago"
    
    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)} hour{'s' if int(hours) != 1 else ''} ago"
    
    days = hours / 24
    if days < 30:
        return f"{int(days)} day{'s' if int(days) != 1 else ''} ago"
    
    months = days / 30
    if months < 12:
        return f"{int(months)} month{'s' if int(months) != 1 else ''} ago"
    
    years = months / 12
    return f"{int(years)} year{'s' if int(years) != 1 else ''} ago"


def start_of_day(dt: Optional[datetime] = None) -> datetime:
    """
    Get start of day (00:00:00) for a given datetime.
    
    Args:
        dt: Input datetime (uses current UTC if None)
        
    Returns:
        datetime: Start of day datetime
    """
    if dt is None:
        dt = utc_now()
    
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def end_of_day(dt: Optional[datetime] = None) -> datetime:
    """
    Get end of day (23:59:59.999999) for a given datetime.
    
    Args:
        dt: Input datetime (uses current UTC if None)
        
    Returns:
        datetime: End of day datetime
    """
    if dt is None:
        dt = utc_now()
    
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


def days_between(start_date: datetime, end_date: datetime) -> int:
    """
    Calculate number of days between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        int: Number of days between dates
    """
    return (end_date.date() - start_date.date()).days


def is_same_day(dt1: datetime, dt2: datetime) -> bool:
    """
    Check if two datetimes are on the same day.
    
    Args:
        dt1: First datetime
        dt2: Second datetime
        
    Returns:
        bool: True if both datetimes are on the same day
    """
    return dt1.date() == dt2.date()


def add_business_days(dt: datetime, days: int) -> datetime:
    """
    Add business days to a datetime (skipping weekends).
    
    Args:
        dt: Starting datetime
        days: Number of business days to add
        
    Returns:
        datetime: Resulting datetime
    """
    current = dt
    days_added = 0
    
    while days_added < days:
        current += timedelta(days=1)
        # Skip weekends (Saturday = 5, Sunday = 6)
        if current.weekday() < 5:
            days_added += 1
    
    return current


def get_week_boundaries(dt: Optional[datetime] = None) -> tuple[datetime, datetime]:
    """
    Get start and end of week (Monday to Sunday) for a given datetime.
    
    Args:
        dt: Input datetime (uses current UTC if None)
        
    Returns:
        tuple: (start_of_week, end_of_week)
    """
    if dt is None:
        dt = utc_now()
    
    # Monday is 0, Sunday is 6
    days_since_monday = dt.weekday()
    
    start_of_week = start_of_day(dt - timedelta(days=days_since_monday))
    end_of_week = end_of_day(start_of_week + timedelta(days=6))
    
    return start_of_week, end_of_week


def get_month_boundaries(dt: Optional[datetime] = None) -> tuple[datetime, datetime]:
    """
    Get start and end of month for a given datetime.
    
    Args:
        dt: Input datetime (uses current UTC if None)
        
    Returns:
        tuple: (start_of_month, end_of_month)
    """
    if dt is None:
        dt = utc_now()
    
    # Start of month
    start_of_month = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # End of month
    if dt.month == 12:
        next_month = dt.replace(year=dt.year + 1, month=1, day=1)
    else:
        next_month = dt.replace(month=dt.month + 1, day=1)
    
    end_of_month = next_month - timedelta(microseconds=1)
    
    return start_of_month, end_of_month


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time if self.start_time else 0
        return self.end_time - self.start_time
    
    @property
    def elapsed_human(self) -> str:
        """Get elapsed time in human-readable format."""
        return seconds_to_human(self.elapsed)
    
    def log_elapsed(self, logger_instance: Optional[logging.Logger] = None):
        """Log the elapsed time."""
        if logger_instance is None:
            logger_instance = logger
        
        logger_instance.info(f"{self.description} completed in {self.elapsed_human}")


class RateLimiter:
    """Simple rate limiter for API calls or operations."""
    
    def __init__(self, max_calls: int, time_window: int):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def can_proceed(self) -> bool:
        """
        Check if another call can proceed without hitting rate limit.
        
        Returns:
            bool: True if call can proceed
        """
        now = time.time()
        
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        return len(self.calls) < self.max_calls
    
    def record_call(self):
        """Record a call for rate limiting."""
        self.calls.append(time.time())
    
    def wait_time(self) -> float:
        """
        Get time to wait before next call can proceed.
        
        Returns:
            float: Seconds to wait (0 if can proceed immediately)
        """
        if self.can_proceed():
            return 0
        
        now = time.time()
        oldest_call = min(self.calls)
        return self.time_window - (now - oldest_call)
    
    def __enter__(self):
        """Context manager entry - wait if necessary."""
        wait_time = self.wait_time()
        if wait_time > 0:
            time.sleep(wait_time)
        self.record_call()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass