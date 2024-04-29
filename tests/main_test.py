import datetime
import pytest

from finra_trio import main


def test_date_range_enforces_start_before_end():
    with pytest.raises(ValueError):
        main.DateRange(start=datetime.date(2024, 1, 1), end=datetime.date(2023, 1, 1))


def test_date_range_equivalence():
    a = main.DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 1, 2))
    b = main.DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 1, 2))

    assert a == b
    # should only be able to compare to other DateRange
    assert not a == datetime.date(2024, 1, 1)


def test_date_range_ordering():
    a = main.DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 1, 2))
    b = main.DateRange(datetime.date(2024, 1, 2), datetime.date(2024, 1, 3))

    assert a < b
    assert a < datetime.date(2024, 1, 2)
    assert a < datetime.datetime(2024, 1, 2)

    with pytest.raises(TypeError):
        a < 1


def test_date_range_delta():
    a = main.DateRange(datetime.date(2024, 1, 1), datetime.date(2024, 1, 5))
    assert a.delta == datetime.timedelta(days=4)