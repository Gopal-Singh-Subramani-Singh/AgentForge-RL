
import importlib
import pytest
import app


def setup_function():
    importlib.reload(app)


def test_default_balance():
    account = app.BankAccount("Alice")
    assert account.get_balance() == 0


def test_initial_balance():
    account = app.BankAccount("Alice", 100)
    assert account.get_balance() == 100


def test_owner():
    account = app.BankAccount("Alice", 50)
    assert account.owner == "Alice"


def test_deposit():
    account = app.BankAccount("Alice", 100)
    result = account.deposit(50)

    assert result == 150
    assert account.get_balance() == 150


def test_withdraw():
    account = app.BankAccount("Alice", 100)
    result = account.withdraw(40)

    assert result == 60
    assert account.get_balance() == 60


def test_negative_initial_balance():
    with pytest.raises(ValueError):
        app.BankAccount("Alice", -1)


def test_zero_deposit():
    account = app.BankAccount("Alice")

    with pytest.raises(ValueError):
        account.deposit(0)


def test_negative_deposit():
    account = app.BankAccount("Alice")

    with pytest.raises(ValueError):
        account.deposit(-10)


def test_zero_withdraw():
    account = app.BankAccount("Alice", 100)

    with pytest.raises(ValueError):
        account.withdraw(0)


def test_negative_withdraw():
    account = app.BankAccount("Alice", 100)

    with pytest.raises(ValueError):
        account.withdraw(-10)


def test_overdraft():
    account = app.BankAccount("Alice", 50)

    with pytest.raises(ValueError):
        account.withdraw(100)


def test_full_withdrawal():
    account = app.BankAccount("Alice", 50)

    assert account.withdraw(50) == 0
