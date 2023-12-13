import os
import sys
import logging
import argparse


from typing import Union


from rag_valuation import utils
from rag_valuation.logger import eval_logger
from rag_valuation.scripts import generate_rag_contexts
from rag_valuation.generate import generate
from rag_valuation.grading import grading

def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--tasks",
        default=None,
        help="To get full list of tasks, use the command lm-eval --tasks list",
    )
    parser.add_argument(   
        "--generate_question_contexts",
        action="store_true",
        default=False,
        help="If True, generate RAG contexts for each task.",
    )
    parser.add_argument(
        "--respond_with_searcher",
        action="store_true",
        default=False,
        help="If True, evaluate with searcher/rag context results.",
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
    parser.add_argument(
        "--grade_responses",
        action="store_true",
        default=False,
        help="If True, grade responses for each task. - Requires that responses have been generated.",
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
    elif args.grade_responses:
        eval_logger.info(f"Grading responses for {args.tasks}")
        # def run(generated_answers, correct_answers, output_path):

        grading.run(
            f"rag_valuation/data/{args.tasks}_baseline_responses.csv",
            f"rag_valuation/data/{args.tasks}_baseline_questions.jsonl",
            f"rag_valuation/data/{args.tasks}_baseline_responses_graded.csv",
        )

    else:
        # todo: support multiple tasks
        task_name = args.tasks
        eval_logger.info(f"Selected Tasks: {task_name}")

        task_rag_contexts_path = f"rag_valuation/data/{task_name}_rag_contexts.jsonl"
        if not os.path.exists(task_rag_contexts_path):
            eval_logger.error(
                f"RAG contexts for task {task_name} not found. Please generate them first."
            )
            sys.exit()
        
        # open the jsonl file, load lines into a list
        with open(task_rag_contexts_path, "r") as f:
            lines = f.readlines()

        if args.eval_with_searcher:
            generate.run_with_searcher(lines, args.rag_csv_path, args.rag_embeddings_path)
        else:
            generate.run(lines)






  
if __name__ == "__main__":
    cli_evaluate()