from app.behavioral_baseline import BehavioralBaseline


baseline = BehavioralBaseline()


# Normal activity window 1
baseline.add_features({
    "user_id": "EMP_001",
    "total_events": 20,
    "file_access_count": 10,
    "file_copy_count": 2,
    "usb_connection_count": 0,
    "data_transfer_count": 1,
    "transfer_volume": 400,
    "after_hours_count": 0,
})


# Normal activity window 2
baseline.add_features({
    "user_id": "EMP_001",
    "total_events": 22,
    "file_access_count": 12,
    "file_copy_count": 1,
    "usb_connection_count": 0,
    "data_transfer_count": 1,
    "transfer_volume": 450,
    "after_hours_count": 1,
})


# Normal activity window 3
baseline.add_features({
    "user_id": "EMP_001",
    "total_events": 21,
    "file_access_count": 11,
    "file_copy_count": 2,
    "usb_connection_count": 1,
    "data_transfer_count": 2,
    "transfer_volume": 500,
    "after_hours_count": 0,
})


print("\n==============================")
print("EMPLOYEE BEHAVIORAL BASELINE")
print("==============================")

result = baseline.calculate_baseline("EMP_001")

for feature, stats in result.items():
    print(f"\n{feature}")
    print(f"  mean    : {stats['mean']:.2f}")
    print(f"  std     : {stats['std']:.2f}")
    print(f"  samples : {stats['samples']}")