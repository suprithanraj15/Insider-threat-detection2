import unittest

from src.db.init_db import SCHEMA_SQL


class TestDatabaseSchema(unittest.TestCase):
    def test_live_processing_table_is_defined(self):
        self.assertIn("CREATE TABLE IF NOT EXISTS live_event_processing", SCHEMA_SQL)
        self.assertIn("CREATE TABLE IF NOT EXISTS live_event_assessments", SCHEMA_SQL)
        self.assertIn("CREATE TABLE IF NOT EXISTS security_events", SCHEMA_SQL)
        self.assertIn("CREATE TABLE IF NOT EXISTS security_alerts", SCHEMA_SQL)

    def test_system_event_unique_index_is_defined(self):
        self.assertIn("idx_security_events_system_source", SCHEMA_SQL)
        self.assertIn("WHERE source_event_id LIKE 'System-%'", SCHEMA_SQL)

    def test_assessment_unique_event_id(self):
        self.assertIn("event_id BIGINT NOT NULL UNIQUE", SCHEMA_SQL)


if __name__ == "__main__":
    unittest.main()
