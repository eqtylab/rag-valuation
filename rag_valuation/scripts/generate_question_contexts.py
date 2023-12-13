import sys
import ast
import json
import datasets
# from datasets.utils import DownloadError

from rag_valuation import utils
from rag_valuation.logger import eval_logger

def run(args):
    eval_logger.info(f"Generating RAG contexts for {args.tasks}")

    if args.rag_csv_path is None:
        eval_logger.error(
            "Must provide a RAG csv path."
        )
        sys.exit()
    if args.rag_embedding_model is None:
        eval_logger.error(
            "Must provide a RAG embedding model to generate contexts (required to encode questions/labels)."
        )
        sys.exit()
    if args.rag_topk is None:
        eval_logger.error(
            "Must provide a RAG topk value."
        )
        sys.exit()
    if args.rag_embeddings_path is None:
        # warn that embeddings will be created, and this could take awhile
        eval_logger.warning(
            "No RAG embeddings path provided, will be computed in process, this could take awhile."
        )


    # for now, support single task

    # first, load the task yaml file from rag_valuation/tasks/{task}/{task}.yaml
    
    task = args.tasks # todo: support multiple tasks
    task_dict = utils.load_yaml_config(f"rag_valuation/tasks/{task}/{task}.yaml")
    print(task_dict)
    dataset_name = task_dict["dataset_path"]

    try:
        dataset = datasets.load_dataset(
                dataset_name,
                data_dir="rag_valuation/data",
            )
    except:
        eval_logger.error(
                f"Dataset {dataset_name} does not exist or there was a download error. Ensure it exists in datasets."
        )
        sys.exit()

    # todo: allow merge of train and validate into test split
    # eval_logger.info(f"Available splits for {dataset_name}: {dataset.keys()}")
    # for now only using test split

    if task_dict["process_docs"] is not None:
        dataset = task_dict["process_docs"](dataset['test'])

    eval_logger.info(f"Number of examples in test split: {len(dataset)}")

    # create a new jsonl file for each task
    # each line in the jsonl file should be a dictionary with the following keys:
    # question, choices, correct_choice_index, correct_choice_text, correct_choice_letter, context

    file_name = f"rag_valuation/data/{task}_rag_contexts.jsonl"
    eval_logger.info(f"Writing RAG contexts to {file_name}")

    with open(file_name, "w") as f:
        # iterate through each example in the dataset
        for example in dataset:

            question_string = utils.apply_template(task_dict["doc_to_text"], example)
            choices = ast.literal_eval(utils.apply_template(task_dict["doc_to_choice"], example))
            correct_choice_index = utils.apply_template(task_dict["doc_to_target"],  example)


            final_question_string = question_string + "\n"

            for i, choice in enumerate(choices):
                final_question_string += f"({chr(i+65)}) {choice} " 

            if correct_choice_index.isdigit():
                correct_choice_index = ast.literal_eval(correct_choice_index)    
            
            correct_choice_letter = chr(int(correct_choice_index)+65)
            correct_choice_text = choices[correct_choice_index]

            json_obj = {
            "question": final_question_string,
            "choices": choices,
            "correct_choice_index": correct_choice_index,
            "correct_choice_text": correct_choice_text,
            "correct_choice_letter": correct_choice_letter
            }

            # Write the JSON object as a string in a single line
            f.write(json.dumps(json_obj) + "\n")

            # save as newline in jsonl file
            # f.write(f"{{\"question\": \"{final_question_string}\", \"choices\": {choices}, \"correct_choice_index\": {correct_choice_index}, \"correct_choice_text\": \"{correct_choice_text}\", \"correct_choice_letter\": \"{correct_choice_letter}\"}}\n")



        


    
    






    

     
    

    



    
