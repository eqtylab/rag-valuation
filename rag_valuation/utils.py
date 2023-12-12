import os
import yaml
import re
import importlib.util
from jinja2 import BaseLoader, Environment, StrictUndefined

def regex_replace(string, pattern, repl, count: int = 0):
    """Implements the `re.sub` function as a custom Jinja filter."""
    return re.sub(pattern, repl, string, count=count)


env = Environment(loader=BaseLoader, undefined=StrictUndefined)
env.filters["regex_replace"] = regex_replace


def apply_template(template: str, doc: dict) -> str:
    rtemplate = env.from_string(template)
    return rtemplate.render(**doc)

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

def import_function(loader, node):
    function_name = loader.construct_scalar(node)
    yaml_path = os.path.dirname(loader.name)

    module_name, function_name = function_name.split(".")
    module_path = os.path.join(yaml_path, "{}.py".format(module_name))

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function = getattr(module, function_name)
    return function


# Add the import_function constructor to the YAML loader
yaml.add_constructor("!function", import_function)


def load_yaml_config(yaml_path=None, yaml_config=None, yaml_dir=None):
    if yaml_config is None:
        with open(yaml_path, "rb") as file:
            yaml_config = yaml.full_load(file)

    if yaml_dir is None:
        yaml_dir = os.path.dirname(yaml_path)

    assert yaml_dir is not None

    if "include" in yaml_config:
        include_path = yaml_config["include"]
        del yaml_config["include"]

        if type(include_path) == str:
            include_path = [include_path]

        # Load from the last one first
        include_path.reverse()
        final_yaml_config = {}
        for path in include_path:
            # Assumes that path is a full path.
            # If not found, assume the included yaml
            # is in the same dir as the original yaml
            if not os.path.isfile(path):
                path = os.path.join(yaml_dir, path)

            try:
                included_yaml_config = load_yaml_config(path)
                final_yaml_config.update(included_yaml_config)
            except Exception as ex:
                # If failed to load, ignore
                raise ex

        final_yaml_config.update(yaml_config)
        return final_yaml_config
    return yaml_config