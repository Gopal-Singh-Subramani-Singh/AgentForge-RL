from solution import fibonacci
import pytest


def test_zero():
    assert fibonacci(0) == 0


def test_one():
    assert fibonacci(1) == 1


def test_two():
    assert fibonacci(2) == 1


def test_five():
    assert fibonacci(5) == 5


def test_ten():
    assert fibonacci(10) == 55


def test_negative():
    with pytest.raises(ValueError):
        fibonacci(-1)
