class BankAccount:
    def __init__(self, owner, initial_balance=0):
        self.owner = owner
        self.balance = initial_balance
        self.transaction_history = []

    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Owner must be a non-empty string.")
        self._owner = value

    @property
    def balance(self):
        return self._balance

    @balance.setter
    def balance(self, value):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("Balance must be a non-negative number.")
        self._balance = value

    @property
    def transaction_history(self):
        return self._transaction_history

    @transaction_history.setter
    def transaction_history(self, value):
        if not isinstance(value, list):
            raise ValueError("Transaction history must be a list.")
        self._transaction_history = value

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit amount must be greater than 0.")
        self.balance += amount
        self.transaction_history.append(f"Deposited {amount}")

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("Withdrawal amount must be greater than 0.")
        if amount > self.balance:
            raise ValueError("Insufficient funds.")
        self.balance -= amount
        self.transaction_history.append(f"Withdrew {amount}")

    def transfer(self, other_account, amount):
        if amount <= 0:
            raise ValueError("Transfer amount must be greater than 0.")
        if amount > self.balance:
            raise ValueError("Insufficient funds.")
        self.balance -= amount
        other_account.balance += amount
        self.transaction_history.append(f"Transferred {amount} to {other_account.owner}")
        other_account.transaction_history.append(f"Received {amount} from {self.owner}")