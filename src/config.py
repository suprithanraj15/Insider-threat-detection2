from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw CERT r4.1 dataset
RAW_DATA = PROJECT_ROOT / "CERT_Dataset" / "r4.1"

# Processed data
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

# Final outputs
OUTPUT_DATA = PROJECT_ROOT / "data" / "outputs"

# Raw CERT files
DEVICE_FILE = RAW_DATA / "device.csv"
EMAIL_FILE = RAW_DATA / "email.csv"
FILE_FILE = RAW_DATA / "file.csv"
HTTP_FILE = RAW_DATA / "http.csv"
LOGON_FILE = RAW_DATA / "logon.csv"
PSYCHOMETRIC_FILE = RAW_DATA / "psychometric.csv"

# LDAP directory
LDAP_DIR = RAW_DATA / "LDAP"