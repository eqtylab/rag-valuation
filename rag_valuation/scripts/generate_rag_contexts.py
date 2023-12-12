import sys
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



    # print available splits
    
    eval_logger.info(f"Number of examples in test split: {len(dataset)}")






    

     
    

    



    
