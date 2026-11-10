import pytest

@pytest.fixture(autouse=True)
def _devteam_isolated_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
import pytest
from expense_tracker.expenses import Expense

def test_expense_creation():
    expense = Expense(100.5, 'Groceries', 'Weekly grocery shopping')
    assert expense.amount == 100.5
    assert expense.category == 'Groceries'
    assert expense.description == 'Weekly grocery shopping'
    assert isinstance(expense.date, str)

def test_expense_to_dict():
    expense = Expense(100.5, 'Groceries', 'Weekly grocery shopping')
    expected_dict = {'amount': 100.5, 'category': 'Groceries', 'description': 'Weekly grocery shopping', 'date': expense.date}
    assert expense.to_dict() == expected_dict

def test_expense_from_dict():
    expense_dict = {'amount': 100.5, 'category': 'Groceries', 'description': 'Weekly grocery shopping', 'date': '2023-04-01 12:00:00'}
    expense = Expense.from_dict(expense_dict)
    assert expense.amount == 100.5
    assert expense.category == 'Groceries'
    assert expense.description == 'Weekly grocery shopping'
    assert expense.date == '2023-04-01 12:00:00'
