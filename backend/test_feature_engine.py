
from app.feature_engine import extract_features
from app.schemas.event import ActivityEvent, EventType


events = [
    ActivityEvent(
        event_id="1",
        user_id="EMP_001",
        timestamp="2026-09-03T10:00:00",
        event_type=EventType.LOGIN,
        action="login",
    ),

    ActivityEvent(
        event_id="2",
        user_id="EMP_001",
        timestamp="2026-09-03T10:01:00",
        event_type=EventType.FILE_ACCESS,
        action="read",
    ),

    ActivityEvent(
        event_id="3",
        user_id="EMP_001",
        timestamp="2026-09-03T10:02:00",
        event_type=EventType.FILE_ACCESS,
        action="read",
    ),

    ActivityEvent(
        event_id="4",
        user_id="EMP_001",
        timestamp="2026-09-03T10:03:00",
        event_type=EventType.USB_CONNECTED,
        action="connect",
    ),

    ActivityEvent(
        event_id="5",
        user_id="EMP_001",
        timestamp="2026-09-03T10:04:00",
        event_type=EventType.FILE_COPY,
        action="bulk_copy",
        value=500,
    ),

    ActivityEvent(
        event_id="6",
        user_id="EMP_001",
        timestamp="2026-09-03T10:05:00",
        event_type=EventType.DATA_TRANSFER,
        action="large_transfer",
        value=2000,
    ),
]


features = extract_features(events)


print("\n==============================")
print("BEHAVIORAL FEATURE VECTOR")
print("==============================")

for key, value in features.items():
    print(f"{key:25} : {value}")