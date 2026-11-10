import pytest

@pytest.fixture(autouse=True)
def _devteam_isolated_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
import pytest
import subprocess
import tempfile

def test_cli_add_expense():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        filename = temp_file.name
        result = subprocess.run(['python', '-m', 'expense_tracker.cli', 'add', '100.5', 'Groceries', 'Weekly grocery shopping', '--filename', filename], capture_output=True, text=True)
        assert result.returncode == 0
        assert 'Added expense' in result.stdout

def test_cli_list_expenses():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        filename = temp_file.name
        subprocess.run(['python', '-m', 'expense_tracker.cli', 'add', '100.5', 'Groceries', 'Weekly grocery shopping', '--filename', filename], capture_output=True, text=True)
        result = subprocess.run(['python', '-m', 'expense_tracker.cli', 'list', '--filename', filename], capture_output=True, text=True)
        assert result.returncode == 0
        assert 'Weekly grocery shopping' in result.stdout

def test_cli_calculate_total():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        filename = temp_file.name
        subprocess.run(['python', '-m', 'expense_tracker.cli', 'add', '100.5', 'Groceries', 'Weekly grocery shopping', '--filename', filename], capture_output=True, text=True)
        result = subprocess.run(['python', '-m', 'expense_tracker.cli', 'total', '--filename', filename], capture_output=True, text=True)
        assert result.returncode == 0
        assert 'Total Spending: 100.5' in result.stdout
