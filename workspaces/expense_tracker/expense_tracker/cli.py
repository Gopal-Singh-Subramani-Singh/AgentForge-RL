import argparse
from .expenses import ExpenseManager

def main():
    parser = argparse.ArgumentParser(description='Expense Tracker CLI')
    subparsers = parser.add_subparsers(dest='command')

    # Add expense
    add_parser = subparsers.add_parser('add', help='Add a new expense')
    add_parser.add_argument('amount', type=float, help='Amount of the expense')
    add_parser.add_argument('category', help='Category of the expense')
    add_parser.add_argument('description', help='Description of the expense')
    add_parser.add_argument('--filename', default='expenses.json', help='Filename to save/load data to/from')

    # List expenses
    list_parser = subparsers.add_parser('list', help='List all expenses')
    list_parser.add_argument('--filename', default='expenses.json', help='Filename to save/load data to/from')

    # Calculate total spending
    total_parser = subparsers.add_parser('total', help='Calculate total spending')
    total_parser.add_argument('--filename', default='expenses.json', help='Filename to save/load data to/from')

    # Save data
    save_parser = subparsers.add_parser('save', help='Save data to a JSON file')
    save_parser.add_argument('filename', help='Filename to save data to')
    save_parser.add_argument('--filename', default='expenses.json', help='Filename to save/load data to/from')

    # Load data
    load_parser = subparsers.add_parser('load', help='Load data from a JSON file')
    load_parser.add_argument('filename', help='Filename to load data from')
    load_parser.add_argument('--filename', default='expenses.json', help='Filename to save/load data to/from')

    args = parser.parse_args()

    expense_manager = ExpenseManager(args.filename)

    if args.command == 'add':
        expense_manager.add_expense(args.amount, args.category, args.description)
    elif args.command == 'list':
        if expense_manager.expenses:
            expense_manager.list_expenses()
        else:
            print('No expenses found.')
    elif args.command == 'total':
        if expense_manager.expenses:
            print(f'Total Spending: {expense_manager.calculate_total()}')
        else:
            print('No expenses found.')
    elif args.command == 'save':
        expense_manager.save_data(args.filename)
    elif args.command == 'load':
        expense_manager.load_data(args.filename)

if __name__ == '__main__':
    main()