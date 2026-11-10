# Bank Account Module

## Description

This Python module provides a simple implementation of a bank account using the `BankAccount` class. The class allows you to create an account, deposit money, withdraw money, and check your balance. All operations are validated to ensure they adhere to standard banking rules.

## Features

- **Constructor**: Initializes a bank account with an owner's name and an optional initial balance (default is 0).
- **deposit(amount)**: Adds a specified amount to the account balance. Raises a `ValueError` if the amount is non-positive.
- **withdraw(amount)**: Subtracts a specified amount from the account balance. Raises a `ValueError` if the amount is non-positive or exceeds the available balance.
- **get_balance()**: Returns the current balance of the account.
- **Invalid Operations**: Raises a `ValueError` for any invalid operation such as negative initial balances, zero or negative deposits/withdrawals, and overdrafts.

## Installation Instructions

No installation is required as the module is self-contained within the `app.py` file. Simply include it in your Python environment.

## How to Run the Application

To use the `BankAccount` class, simply import it into your Python script:

```python
from app import BankAccount

# Create a bank account
account = BankAccount("Alice", 100)

# Deposit money
new_balance = account.deposit(50)
print(f"New Balance: {new_balance}")

# Withdraw money
new_balance = account.withdraw(40)
print(f"New Balance: {new_balance}")

# Check balance
current_balance = account.get_balance()
print(f"Current Balance: {current_balance}")
```

## API or Usage Documentation

### Class: `BankAccount`

#### Constructor

```python
__init__(owner, initial_balance=0)
```

- **Parameters**:
  - `owner`: A string representing the owner of the bank account.
  - `initial_balance`: An optional integer representing the initial balance of the account (default is 0).

- **Raises**:
  - `ValueError`: If the `initial_balance` is negative.

#### Method: `deposit(amount)`

```python
deposit(amount)
```

- **Parameters**:
  - `amount`: A positive integer representing the amount to deposit.

- **Returns**:
  - The new balance after the deposit.

- **Raises**:
  - `ValueError`: If the `amount` is non-positive.

#### Method: `withdraw(amount)`

```python
withdraw(amount)
```

- **Parameters**:
  - `amount`: A positive integer representing the amount to withdraw.

- **Returns**:
  - The new balance after the withdrawal.

- **Raises**:
  - `ValueError`: If the `amount` is non-positive or exceeds the available balance.

#### Method: `get_balance()`

```python
get_balance()
```

- **Returns**:
  - The current balance of the account.

## Example Requests

Here are some example requests demonstrating how to use the `BankAccount` class:

```python
from app import BankAccount

# Create a bank account
account = BankAccount("Alice", 100)

# Deposit money
new_balance = account.deposit(50)
print(f"New Balance: {new_balance}")  # Output: New Balance: 150

# Withdraw money
new_balance = account.withdraw(40)
print(f"New Balance: {new_balance}")  # Output: New Balance: 110

# Check balance
current_balance = account.get_balance()
print(f"Current Balance: {current_balance}")  # Output: Current Balance: 110
```

This README provides a comprehensive overview of the `BankAccount` module, including its features, installation instructions, usage examples, and error handling.