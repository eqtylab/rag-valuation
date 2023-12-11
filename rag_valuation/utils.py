import logging

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
eval_logger = logging.getLogger("rag-valuation")


def get_all_tasks():
    """
    reads all tasks from the tasks directory, returns list of each directory in tasks/
    ... each directory is a task
    """
    import os

    tasks = []
    for task in os.listdir("rag_valuation/tasks"):
        if os.path.isdir(os.path.join("rag_valuation/tasks", task)):
            tasks.append(task)
    return tasks