from datetime import datetime
from enum import Enum

from pydantic import BaseModel
from pydantic import BaseModel, Field


class EventType(str, Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    FILE_ACCESS = "file_access"
    FILE_COPY = "file_copy"
    USB_CONNECTED = "usb_connected"
    WEB_ACTIVITY = "web_activity"
    DATA_TRANSFER = "data_transfer"
    EMAIL_ACTIVITY = "email_activity"


class ActivityEvent(BaseModel):
    event_id: str
    user_id: str
    timestamp: datetime
    event_type: EventType

    resource: str | None = None
    action: str | None = None

    value: float | None = None
    metadata: dict = Field(default_factory=dict)