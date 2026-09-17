from collections import Counter
from typing import List

from app.schemas.event import ActivityEvent


def extract_features(events: List[ActivityEvent]) -> dict:
    """
    Convert a collection of ActivityEvent objects
    into a behavioral feature vector.
    """

    if not events:
        return {}

    user_id = events[0].user_id

    # Count event types
    event_counts = Counter(event.event_type.value for event in events)

    # Count actions
    action_counts = Counter(
        event.action for event in events
        if event.action is not None
    )

    # Calculate total numeric value
    total_value = sum(
        event.value or 0
        for event in events
    )

    features = {
        # User
        "user_id": user_id,

        # Overall activity
        "total_events": len(events),

        # Event-type features
        "login_count": event_counts.get("login", 0),
        "logout_count": event_counts.get("logout", 0),
        "file_access_count": event_counts.get("file_access", 0),
        "file_copy_count": event_counts.get("file_copy", 0),
        "usb_connection_count": event_counts.get("usb_connected", 0),
        "web_activity_count": event_counts.get("web_activity", 0),
        "data_transfer_count": event_counts.get("data_transfer", 0),
        "email_activity_count": event_counts.get("email_activity", 0),

        # Action features
        "read_count": action_counts.get("read", 0),
        "visit_count": action_counts.get("visit", 0),
        "bulk_copy_count": action_counts.get("bulk_copy", 0),
        "large_transfer_count": action_counts.get("large_transfer", 0),

        # Numeric activity
        "total_value": total_value,
    }

    return features 