#!/bin/bash
# Task Manager — Entry point script
# BUG: The Python path is wrong (uses 'python3' but some systems only have 'python')

set -e

echo "=== Task Manager v0.1.0 ==="
echo ""

# BUG: Should check if python3 exists, fall back to python
python3 -c "
from src.manager import TaskManager
from src.formatter import format_task_table, format_stats

manager = TaskManager()

# Add some sample tasks
manager.add_task('Set up CI/CD pipeline', tags=['devops', 'urgent'])
manager.add_task('Write API documentation', description='Document all REST endpoints')
manager.add_task('Fix authentication bug', tags=['bug', 'security'])
manager.add_task('Add unit tests', description='Increase coverage to 80%', tags=['testing'])
manager.add_task('Refactor database layer', tags=['tech-debt'])

# Complete one task
manager.complete_task(3)

# Display results
print(format_task_table(manager.list_tasks()))
print()
print(format_stats(manager.stats()))
print()
print('Searching for \"bug\":')
results = manager.search_tasks('bug')
print(format_task_table(results))
"
