import os
import sys
import logging
import argparse


from typing import Union


from rag_valuation import utils
from rag_valuation.logger import eval_logger
from rag_valuation.scripts import generate_rag_contexts

def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--tasks",
        default=None,
        help="To get full list of tasks, use the command lm-eval --tasks list",
    )
    parser.add_argument(
        "--model_args",
        default="",
        help="String arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of examples in few-shot context",
    )
    parser.add_argument("--batch_size", type=str, default=1)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        metavar="= [dir/file.jsonl] [DIR]",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--use_cache",
        type=str,
        default=None,
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument("--decontamination_ngrams_path", default=None)  # TODO: not used
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks",
    )
    parser.add_argument(
        "--write_out",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        default="",
        help=(
            "String arguments for model generation on greedy_until tasks,"
            " e.g. `temperature=0,top_k=0,top_p=0`"
        ),
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Log error when tasks are not registered.",
    )
    parser.add_argument(   
        "--generate_rag_contexts",
        action="store_true",
        default=False,
        help="If True, generate RAG contexts for each task.",
    )
    parser.add_argument(
        "--rag_csv_path",
        type=str,
        default=None,
        help="Path to CSV containing RAG contexts. Must have column `chunk_text`",
    )
    parser.add_argument(
        "--rag_embedding_model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="RAG model to use for generating contexts.",
    )
    parser.add_argument(
        "--rag_embeddings_path",
        type=str,
        default=None,
        help="Path to (optional) pre-computed RAG embeddings, if not provided, will be computed in process.",
    )
    parser.add_argument(
        "--rag_topk",
        type=int,
        default=100,
        help="Number of retrieved candiates, to be paired N times with each datum the dataset (new_dataset_size = (N*dataset_size))",
    )
    return parser.parse_args()


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        # we allow for args to be passed externally, else we parse them ourselves
        args = parse_eval_args()

    ALL_TASKS = utils.get_all_tasks()

    
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    

    if args.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING."
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

     

    if args.tasks is None:
        # error
        eval_logger.error(
            "No tasks were selected. Try `rag-valuation --tasks list` for list of available tasks."
        )
        sys.exit()
    elif args.tasks == "list":
        eval_logger.info(
            "Available Tasks:\n - {}".format(f"\n - ".join(sorted(ALL_TASKS)))
        )
        sys.exit()

    if args.generate_rag_contexts:
        eval_logger.info(f"Generating RAG contexts for {args.tasks}")

        generate_rag_contexts.run(args)
        sys.exit()
    else:
        
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            tasks_list = args.tasks.split(",")
            task_names = utils.pattern_match(tasks_list, ALL_TASKS)
            for task in [task for task in tasks_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task
                for task in tasks_list
                if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n"
                    f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks {missing} were not found. Try `lm-eval --tasks list` for list of available tasks."
                )


    eval_logger.info(f"Selected Tasks: {task_names}")



  
if __name__ == "__main__":
    cli_evaluate()