
import re


def extract_service_evidence(message: str) -> dict:
    """
    Extract structured service evidence from Windows System event messages.

    Supports:
    - Event ID 7045: Service installation
    - Event ID 7040: Service start-type modification
    """

    text = (message or "").replace("\r\n", "\n").replace("\r", "\n")
    evidence = {}

    field_patterns = {
        "service_name": r"^\s*Service Name:\s*(.*?)\s*$",
        "service_file_name": r"^\s*Service File Name:\s*(.*?)\s*$",
        "service_type": r"^\s*Service Type:\s*(.*?)\s*$",
        "service_start_type": r"^\s*Service Start Type:\s*(.*?)\s*$",
        "service_account": r"^\s*Service Account:\s*(.*?)\s*$",
    }

    for field, pattern in field_patterns.items():
        match = re.search(
            pattern,
            text,
            re.IGNORECASE | re.MULTILINE
        )

        if match:
            value = match.group(1).strip().strip('"')
            if value:
                evidence[field] = value

    start_type_pattern = (
        r"The start type of the\s+(.+?)\s+service was changed "
        r"from\s+(.+?)\s+to\s+(.+?)(?:\.|$)"
    )

    start_type_change = re.search(
        start_type_pattern,
        text,
        re.IGNORECASE | re.DOTALL
    )

    if start_type_change:
        evidence["service_name"] = start_type_change.group(1).strip()
        evidence["previous_start_type"] = start_type_change.group(2).strip()
        evidence["new_start_type"] = start_type_change.group(3).strip()

    return evidence
