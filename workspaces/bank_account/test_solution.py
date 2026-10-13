import pytest

from solution import BankAccount


def test_create_account():
    account = BankAccount("Alice")

    assert account.owner == "Alice"
    assert account.balance == 0
    assert isinstance(account.transaction_history, list)


def test_initial_balance():
    account = BankAccount("Alice", 100)

    assert account.balance == 100


def test_deposit():
    account = BankAccount("Alice", 100)

    account.deposit(50)

    assert account.balance == 150
    assert len(account.transaction_history) == 1


def test_negative_deposit():
    account = BankAccount("Alice", 100)

    with pytest.raises(ValueError):
        account.deposit(-10)

    assert account.balance == 100
    assert len(account.transaction_history) == 0


def test_zero_deposit():
    account = BankAccount("Alice", 100)

    with pytest.raises(ValueError):
        account.deposit(0)


def test_withdraw():
    account = BankAccount("Alice", 100)

    account.withdraw(30)

    assert account.balance == 70
    assert len(account.transaction_history) == 1


def test_overdraft():
    account = BankAccount("Alice", 100)

    with pytest.raises(ValueError):
        account.withdraw(150)

    assert account.balance == 100
    assert len(account.transaction_history) == 0


def test_negative_withdrawal():
    account = BankAccount("Alice", 100)

    with pytest.raises(ValueError):
        account.withdraw(-20)


def test_transfer():
    alice = BankAccount("Alice", 100)
    bob = BankAccount("Bob", 20)

    alice.transfer(bob, 40)

    assert alice.balance == 60
    assert bob.balance == 60

    assert len(alice.transaction_history) == 1
    assert len(bob.transaction_history) == 1


def test_failed_transfer():
    alice = BankAccount("Alice", 20)
    bob = BankAccount("Bob", 50)

    with pytest.raises(ValueError):
        alice.transfer(bob, 100)

    assert alice.balance == 20
    assert bob.balance == 50

    assert len(alice.transaction_history) == 0
    assert len(bob.transaction_history) == 0


def test_zero_transfer():
    alice = BankAccount("Alice", 100)
    bob = BankAccount("Bob", 100)

    with pytest.raises(ValueError):
        alice.transfer(bob, 0)


def test_multiple_transactions():
    account = BankAccount("Alice", 100)

    account.deposit(50)
    account.withdraw(20)
    account.deposit(10)

    assert account.balance == 140
    assert len(account.transaction_history) == 3
