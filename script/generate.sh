# every task, max 20k
python -m task.run --splits train --max_instance_per_task 20000

# for test, only klue and kobest, all items
python -m task.run --splits test --tasks "nsmc,apeach,klue_*,kobest_*,haerae_csatqa" --max_instance_per_task -1

python -m task.push_to_hub iknow-lab/koflan-0813