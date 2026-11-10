from datetime import datetime
import json
class Expense:
    def __init__(self, amount, category, description, date=None):
        self.amount = amount
        self.category = category
        self.description = description
        self.date = date if date else datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def to_dict(self):
        return {
            'amount': self.amount,
            'category': self.category,
            'description': self.description,
            'date': self.date
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data['amount'], data['category'], data['description'], data.get('date'))
class ExpenseManager:
    def __init__(self, filename='expenses.json'):
        self.filename = filename
        self.expenses = self.load_data()

    def add_expense(self, amount, category, description):
        expense = Expense(amount, category, description)
        self.expenses.append(expense)
        self.save_data()

    def list_expenses(self):
        if self.expenses:
            for expense in self.expenses:
                print(f'{expense.date} - {expense.category}: {expense.description} (${expense.amount})')
        else:
            print('No expenses found.')

    def calculate_total(self):
        return sum(expense.amount for expense in self.expenses) if self.expenses else 0

    def save_data(self):
        with open(self.filename, 'w') as f:
            json.dump([expense.to_dict() for expense in self.expenses], f, indent=4)

    def load_data(self):
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
                return [Expense.from_dict(item) for item in data]
        except FileNotFoundError:
            return []