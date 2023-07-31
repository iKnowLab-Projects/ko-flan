# every task, max 20k
python -m task.run --splits train --max_instance_per_task 20000

# for test, only klue and kobest, all items
python -m task.run --splits test --tasks "klue_*,kobest_*" --max_instance_per_task -1