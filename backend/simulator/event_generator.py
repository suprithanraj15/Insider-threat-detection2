import random
import uuid
from datetime import datetime, timedelta

from app.schemas.event import ActivityEvent, EventType


def generate_normal_event(user_id: str, timestamp: datetime) -> ActivityEvent:
    event_type = random.choice(
        [
            EventType.LOGIN,
            EventType.FILE_ACCESS,
            EventType.WEB_ACTIVITY,
            EventType.LOGOUT,
        ]
    )

    if event_type == EventType.LOGIN:
        return ActivityEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            timestamp=timestamp,
            event_type=event_type,
            action="login",
        )

    if event_type == EventType.FILE_ACCESS:
        return ActivityEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            timestamp=timestamp,
            event_type=event_type,
            resource="project_document.pdf",
            action="read",
            value=1,
        )

    if event_type == EventType.WEB_ACTIVITY:
        return ActivityEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            timestamp=timestamp,
            event_type=event_type,
            resource="internal.company.portal",
            action="visit",
        )

    return ActivityEvent(
        event_id=str(uuid.uuid4()),
        user_id=user_id,
        timestamp=timestamp,
        event_type=EventType.LOGOUT,
        action="logout",
    )


def generate_suspicious_event(
    user_id: str,
    timestamp: datetime,
) -> ActivityEvent:

    event_type = random.choice(
        [
            EventType.FILE_ACCESS,
            EventType.FILE_COPY,
            EventType.USB_CONNECTED,
            EventType.DATA_TRANSFER,
        ]
    )

    if event_type == EventType.FILE_ACCESS:
        return ActivityEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            timestamp=timestamp,
            event_type=event_type,
            resource="sensitive_test_document.zip",
            action="read",
            value=random.randint(50, 200),
            metadata={
                "sensitive": True,
                "bulk_access": True,
            },
        )

    if event_type == EventType.FILE_COPY:
        return ActivityEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            timestamp=timestamp,
            event_type=event_type,
            resource="test_sensitive_directory",
            action="bulk_copy",
            value=random.randint(100, 500),
            metadata={
                "sensitive": True,
                "bulk_operation": True,
            },
        )

    if event_type == EventType.USB_CONNECTED:
        return ActivityEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            timestamp=timestamp,
            event_type=event_type,
            action="connect",
            metadata={
                "new_device": True,
            },
        )

    return ActivityEvent(
        event_id=str(uuid.uuid4()),
        user_id=user_id,
        timestamp=timestamp,
        event_type=EventType.DATA_TRANSFER,
        action="large_transfer",
        value=random.uniform(500, 5000),
        metadata={
            "external_destination": True,
        },
    )