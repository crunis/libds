import unittest
from datetime import datetime, timedelta
from libds.misc.group import week, group


class TestGroup(unittest.TestCase):

    def test_week_within_week(self):
        dts = [datetime(2023, 10, 26)]
        dt = datetime(2023, 10, 30)
        self.assertTrue(week(dts, dt))

    def test_week_exactly_one_week_later(self):
        dts = [datetime(2023, 10, 26)]
        dt = datetime(2023, 11, 1)
        self.assertTrue(week(dts, dt))

    def test_week_just_over_one_week(self):
        dts = [datetime(2023, 10, 26)]
        dt = datetime(2023, 11, 2)
        self.assertFalse(week(dts, dt))

    def test_week_same_day(self):
        dts = [datetime(2023, 10, 26)]
        dt = datetime(2023, 10, 26)
        self.assertTrue(week(dts, dt))

    def test_week_multiple_dates(self):
        dts = [datetime(2023, 10, 20), datetime(2023, 10, 25), datetime(2023, 10, 26)]
        dt = datetime(2023, 10, 30)
        self.assertTrue(week(dts, dt))

    def test_week_multiple_dates_over_week(self):
        dts = [datetime(2023, 10, 20), datetime(2023, 10, 25), datetime(2023, 10, 26)]
        dt = datetime(2023, 11, 3)
        self.assertFalse(week(dts, dt))

    def test_group_empty_list(self):
        self.assertEqual(group([]), [])

    def test_group_single_item(self):
        dts = [datetime(2023, 10, 26)]
        self.assertEqual(group(dts), [[datetime(2023, 10, 26)]])

    def test_group_within_week(self):
        dts = [datetime(2023, 10, 26), datetime(2023, 10, 27), datetime(2023, 10, 28)]
        self.assertEqual(group(dts), [[datetime(2023, 10, 26), datetime(2023, 10, 27), datetime(2023, 10, 28)]])

    def test_group_over_week(self):
        dts = [datetime(2023, 10, 26), datetime(2023, 11, 2), datetime(2023, 11, 3)]
        expected = [[datetime(2023, 10, 26)], [datetime(2023, 11, 2), datetime(2023, 11, 3)]]
        self.assertEqual(group(dts), expected)

    def test_group_multiple_groups(self):
        dts = [datetime(2023, 10, 26), datetime(2023, 10, 27), datetime(2023, 11, 3), datetime(2023, 11, 4), datetime(2023, 11, 12)]
        expected = [[datetime(2023, 10, 26), datetime(2023, 10, 27)], [datetime(2023, 11, 3), datetime(2023, 11, 4)], [datetime(2023, 11, 12)]]
        self.assertEqual(group(dts), expected)

    def test_group_with_custom_condition(self):
        def custom_condition(group, item):
            return item - group[-1] <= timedelta(days=2)

        dts = [datetime(2023, 10, 26), datetime(2023, 10, 27), datetime(2023, 10, 30), datetime(2023, 10, 31)]
        expected = [[datetime(2023, 10, 26), datetime(2023, 10, 27)], [datetime(2023, 10, 30), datetime(2023, 10, 31)]]
        self.assertEqual(group(dts, condition=custom_condition), expected)

    def test_group_with_numbers(self):
        def custom_condition(group, item):
            return item - group[-1] <= 2
        
        numbers = [1, 2, 3, 6, 7, 8, 11]
        expected = [[1, 2, 3], [6, 7, 8], [11]]
        self.assertEqual(group(numbers, condition=custom_condition), expected)

if __name__ == '__main__':
    unittest.main()
