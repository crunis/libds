import unittest
import pandas as pd
from datetime import datetime
from libds.misc.find_closest_event import find_closest_event


class TestFindClosestEvent(unittest.TestCase):

    def setUp(self):
        data = {
            "pid": [1, 1, 2, 2],
            "_dt": [
                datetime(2023, 1, 1),
                datetime(2023, 1, 5),
                datetime(2023, 1, 3),
                datetime(2023, 1, 7),
            ],
        }
        self.df = pd.DataFrame(data)

    def test_find_closest_event_inclusive(self):
        result = find_closest_event(self.df, 1, datetime(2023, 1, 5), inclusive=True)
        self.assertTrue(result["_"])
        self.assertEqual(result["_days"], 0)

    def test_find_closest_event_exclusive(self):
        result = find_closest_event(self.df, 1, datetime(2023, 1, 5), inclusive=False)
        self.assertTrue(result["_"])
        self.assertEqual(result["_days"], 4)

    def test_find_closest_event_no_event(self):
        result = find_closest_event(self.df, 1, datetime(2022, 12, 31))
        self.assertFalse(result["_"])
        self.assertEqual(result["_days"], None)

    def test_find_closest_event_with_prefix(self):
        result = find_closest_event(self.df, 2, datetime(2023, 1, 4), prefix="test")
        self.assertTrue(result["test_"])
        self.assertEqual(result["test_days"], 1)


if __name__ == "__main__":
    unittest.main()
