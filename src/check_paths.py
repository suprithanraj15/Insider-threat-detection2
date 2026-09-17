from config import (
    RAW_DATA,
    PROCESSED_DATA,
    OUTPUT_DATA,
    DEVICE_FILE,
    EMAIL_FILE,
    FILE_FILE,
    HTTP_FILE,
    LOGON_FILE,
    PSYCHOMETRIC_FILE,
    LDAP_DIR,
)

print("=" * 60)
print("CERT DATASET PATH CHECK")
print("=" * 60)

print(f"\nRaw dataset:")
print(RAW_DATA)
print("Exists:", RAW_DATA.exists())

print(f"\nProcessed folder:")
print(PROCESSED_DATA)
print("Exists:", PROCESSED_DATA.exists())

print(f"\nOutput folder:")
print(OUTPUT_DATA)
print("Exists:", OUTPUT_DATA.exists())

print("\nRaw files:")

files = [
    DEVICE_FILE,
    EMAIL_FILE,
    FILE_FILE,
    HTTP_FILE,
    LOGON_FILE,
    PSYCHOMETRIC_FILE,
]

for file in files:
    print(f"{file.name:20} -> {file.exists()}")

print(f"\nLDAP folder:")
print(LDAP_DIR)
print("Exists:", LDAP_DIR.exists())

print("\n" + "=" * 60)