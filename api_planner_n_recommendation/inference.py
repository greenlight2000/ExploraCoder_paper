import pandas as pd
import numpy as np
import sys
import os
from models import OpenAIEmbModel, OpenAIChatModel
from example_selecters import ExampleSelecter
from decomposers import Decomposer
from retrievers import APIRetriever, build_emb_bank
from rerankers import Reranker
import logging
from utils import setup_log
from tqdm import tqdm


def adaretrieve(results_path):
    retriever.load_apis()
    with open(results_path, 'w') as f:
        f.write('')

    error_index = []
    for (i, item), query_emb in zip(testset.iterrows(), testset_emb_mat):
        item['task_id'] = i
        logging.info("========================================================================")
        logging.info(f"Processing task {i} task_id={item['task_id']}")
        logging.info("========================================================================")
        task = item['task']
        gt_apis = item['apis']
        try:
            apidf = retriever.retrieve_apis([query_emb], k_total_api)[0]
        except Exception as e:
            logging.error(f"Error in processing task {i}:\n{e}")
            results = pd.DataFrame([["", "", ""]],columns=['task', 'gt_apis', 'retrieved_apis'])
            results.to_json(results_path, orient='records', lines=True, mode='a')
            error_index.append(i)
            continue
        results = pd.DataFrame([[
            task,
            gt_apis,
            apidf
        ]],columns=['task', 'gt_apis', 'retrieved_apis'])
        results.to_json(results_path, orient='records', lines=True, mode='a')
        logging.info(f"------------------------------finished task {i}  task_id={item['task_id']}------------------------------")
    if len(error_index) != 0:
        logging.warning(f"Error in processing tasks index={error_index}. please check the log file for details and rerun the corresponding tasks in {testset_path}")
    else:
        logging.info("All tasks processed successfully.")
    return error_index

def decompretrieve_pipeline(results_path):
    es.load_examples()
    retriever.load_apis()
    with open(results_path, 'w') as f:
        f.write('')

    error_index = []
    for i, item in testset.iterrows():
        logging.info("========================================================================")
        logging.info(f"Processing task {i} task_id={item['task_id']}")
        logging.info("========================================================================")
        task = item['task']
        gt_apis = item['gt_apis'] # ["IterableWrapper", "cycle", "Zipper"]

        try:
            logging.info("------------------------------selecting examples------------------------------")
            examples = es.select_examples(task, k_example) # df["task","apis","subtasks"]
            logging.info("------------------------------decomposing task------------------------------")
            subtasks = decomposer.decompose(task, examples) # ["xxx", "xxx"]
            logging.info("------------------------------retrieving apis-----------------------------")
            apidf_per_subtask = retriever.retrieve_apis(subtasks, k_subtask_api)
            logging.info("------------------------------intra reranking------------------------------")
            reranked_apidf_per_subtask = []
            for subtask, api_df in zip(subtasks, apidf_per_subtask):
                reranked_apidf = reranker.intra_subtask_rerank(library=library, subtask=subtask, apidf=api_df, k_api=k_subtask_api)
                reranked_apidf_per_subtask.append(reranked_apidf)
            logging.info("------------------------------inter reranking------------------------------")
            final_apis = reranker.inter_subtask_rerank(library, task, subtasks, reranked_apidf_per_subtask, api_bank, k_subtask_api, k_total_api)
        except Exception as e:
            logging.error(f"Error in processing task {i}:\n{e}")
            results = pd.DataFrame([["", "", "", "", "", "", ""]],columns=['task', 'gt_apis', "selected_examples", 'decomposed_subtasks', 'retrieved_apis_per_subtask', 'reranked_apis_per_subtask', 'final_reranked_apis'])
            results.to_json(results_path, orient='records', lines=True, mode='a')
            error_index.append(i)
            continue
        
        results = pd.DataFrame([[
            task,
            gt_apis,
            examples,
            subtasks,
            [eval(apidf.to_json(orient='records'),{'null':''}) for apidf in apidf_per_subtask],
            [eval(apidf.to_json(orient='records'),{'null':''}) for apidf in reranked_apidf_per_subtask],
            final_apis
        ]],columns=['task', 'gt_apis', "selected_examples", 'decomposed_subtasks', 'retrieved_apis_per_subtask', 'reranked_apis_per_subtask', 'final_reranked_apis'])
        results.to_json(results_path, orient='records', lines=True, mode='a')
        logging.info(f"------------------------------finished task {i} task_id={item['task_id']}------------------------------")
    if len(error_index) != 0:
        logging.warning(f"Error in processing tasks index={error_index}. please check the log file for details and rerun the corresponding tasks in {testset_path}")
    else:
        logging.info(f"Finished inference.")

def decompretrieve_batch(subtask_emb_path, example_subtask_output_path, api_retrieve_rerank_output_path):
    """
    Recommend APIs for each subtask in each task in the testset. The recommendation is two-step:
    First select decomposition examples by comparing the testset index and examplebank index. And then decompose each task in testset into subtasks. Results are stored in `example_subtask_output_path`. And prepare a (task, subtask, index) npy file in `subtask_emb_path` for the next step.
    Second retrieve APIs for each subtask in each task. Results are stored in `api_retrieve_rerank_output_path`.
    `example_subtask_output_path` and `api_retrieve_rerank_output_path` can be the same file, and the results are appended to the file.
    """
    es.load_examples()
    retriever.load_apis()

    # select examples and decompose tasks
    logging.info('=========================================================')
    logging.info('Example Selection and Task Decomposing')
    logging.info('=========================================================')

    if not os.path.exists(subtask_emb_path):
        logging.info('subtask emb not found, start building task subtask embeddings matrix...')
        task_subtask_emb_li = [] # shape: (n_task, n_subtask, emb_dim)
        if not os.path.exists(example_subtask_output_path):
            logging.info('examples and subtasks records not found, start selecting examples and decomposing subtasks, meanwhile an embedding metrix for the subtasks in each task is to be built...')
            for (i, item), task_emb in zip(testset.iterrows(), testset_emb_mat):
                examples = es.select_examples(task_emb, k_example)
                subtasks = decomposer.decompose(item['task'], examples)
                emb_li = []
                for subtask in subtasks:
                    emb = emb_model.get_emb(text=subtask) # shape: (emb_dim,)
                    emb_li.append(emb)
                emb_mat = np.array(emb_li) # shape: (n_subtasks, emb_dim)
                task_subtask_emb_li.append(emb_mat)
                results = pd.DataFrame([[
                    # item['task_id'],
                    item['task'],
                    item['apis'] if 'apis' in item.keys() else item['gt_apis'], #item['apis'] or item['gt_apis']
                    examples,
                    subtasks,
                ]],columns=['task', 'gt_apis', "selected_examples", 'decomposed_subtasks'])
                results.to_json(example_subtask_output_path, orient='records', lines=True, mode='a')
            logging.info(f"successfully select examples and decompose subtasks for testset: `{testset_path}`. The results are stored at `{example_subtask_output_path}`")
        else:
            logging.info('examples and subtasks records found, start loading records and building embeddings matrix for subtasks from it...')
            example_subtask_records = pd.read_json(example_subtask_output_path, lines=True, orient='records')
            for (i, item) in example_subtask_records.iterrows():
                subtasks = item['decomposed_subtasks']
                emb_li = []
                for subtask in subtasks:
                    emb = emb_model.get_emb(text=subtask)
                    emb_li.append(emb)
                emb_mat = np.array(emb_li) # shape: (n_subtasks, emb_dim)
                task_subtask_emb_li.append(emb_mat)
        task_subtask_emb_mat = np.array(task_subtask_emb_li, dtype=object) # shape: (n_task, n_subtask, emb_dim)
        np.save(subtask_emb_path, task_subtask_emb_mat)
        logging.info(f"built task subtask embeddings of shape: `({task_subtask_emb_mat.shape[0]}, n_subtask, {len(emb)})`, at `{subtask_emb_path}`")
    else:
        logging.info(f"found example subtask records, and their corresponding subtask emb metrix. Skip Example Selection and Task Decomposition")
    # retrieve apis
    logging.info('=========================================================')
    logging.info('API Retrival and Reranking')
    logging.info('=========================================================')
    if not os.path.exists(api_recommendation_output_path):
        logging.info('api retrieve rerank records not found, start retrieving apis and reranking...')
        task_subtask_emb_mat = np.load(subtask_emb_path, allow_pickle=True) # batch input
        example_subtask_records = pd.read_json(example_subtask_output_path, lines=True, orient='records') # batch input
        for (i, item), subtask_emb_mat in tqdm(zip(example_subtask_records.iterrows(), task_subtask_emb_mat), total=task_subtask_emb_mat.shape[0]):
            apidf_per_subtask = retriever.retrieve_apis(subtask_emb_mat, k_subtask_api)
            reranked_apidf_per_subtask = []
            subtasks = item['decomposed_subtasks']
            for subtask, api_df in zip(subtasks, apidf_per_subtask):
                reranked_apidf = reranker.intra_subtask_rerank(library=library, subtask=subtask, apidf=api_df, k_api=k_subtask_api)
                reranked_apidf_per_subtask.append(reranked_apidf)
            final_apis = reranker.inter_subtask_rerank(library, item['task'], subtasks, reranked_apidf_per_subtask, api_bank, k_subtask_api, k_total_api)

            gt_apis = item['gt_apis']

            results = pd.DataFrame([[
                # item['task_id'],
                item['task'],
                gt_apis,
                item['selected_examples'],
                subtasks,
                [eval(apidf.to_json(orient='records'),{'null':''}) for apidf in apidf_per_subtask],
                [eval(apidf.to_json(orient='records'),{'null':''}) for apidf in reranked_apidf_per_subtask],
                final_apis
            ]],columns=['task', 'gt_apis', "selected_examples", 'decomposed_subtasks', 'retrieved_apis_per_subtask', 'reranked_apis_per_subtask', 'final_reranked_apis'])
            results.to_json(api_recommendation_output_path, orient='records', lines=True, mode='a')
        logging.info(f"successfully retrieve apis and rerank for testset: `{testset_path}`. The results are stored at `{api_recommendation_output_path}`")
    else:
        logging.info(f"found api retrieve rerank records. Skip API Retrival and Reranking")
    logging.info('===========================finished==============================')


def read_df(path):
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.json'):
        return pd.read_json(path)
    elif path.endswith('.jsonl'):
        return pd.read_json(path, lines=True, orient='records')
    else:
        raise ValueError(f"Unsupported file format: {path}")


import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=str, default="TorchData")
    parser.add_argument('--k_example', type=int, default=4)
    parser.add_argument('--k_subtask_api', type=int, default=15)
    parser.add_argument('--k_total_api', type=int, default=20)
    parser.add_argument('--example_bank_path', type=str)
    parser.add_argument('--examplebank_emb_path', type=str)
    parser.add_argument('--example_emb_field', type=str, default='task')
    parser.add_argument('--api_bank_path', type=str)
    parser.add_argument('--apibank_emb_path', type=str)
    parser.add_argument('--api_emb_field', type=str)
    parser.add_argument('--testset_path', type=str)
    parser.add_argument('--testset_emb_field', type=str, default='task')
    parser.add_argument('--testset_emb_path', type=str, default='./torchdata_manual_task_emb.npy')
    parser.add_argument('--log_path', type=str, help="Path to place the log file")
    parser.add_argument('--emb_model', type=str, default='text-embedding-ada-002')
    parser.add_argument('--gen_model', type=str, default='gpt-3.5-turbo-0125')
    parser.add_argument('--openai_key', type=str, help="OpenAI API key")
    parser.add_argument('--method', type=str, choices=['adaretrieve', 'decompretrieve'])
    parser.add_argument('--api_recommendation_output_path', type=str)

    # for batch processed decompretrieve
    parser.add_argument('--example_subtask_output_path', type=str)
    parser.add_argument('--subtask_emb_path', type=str)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    args = parse_args()
    logger = setup_log(args.log_path, console=False)

    api_bank = read_df(args.api_bank_path)
    k_total_api=args.k_total_api
    emb_model = OpenAIEmbModel(args.openai_key, model_name=args.emb_model)
    retriever = APIRetriever(args.api_bank_path, args.api_emb_field, emb_model, args.apibank_emb_path)
    if not os.path.exists(retriever.apibank_emb_path):
        logging.info('apibank_emb not found.')
        retriever.build_api_emb_pool()
    testset_path = args.testset_path
    testset = read_df(testset_path)
    testset_emb_path = args.testset_emb_path
    if not os.path.exists(testset_emb_path):
        logging.info('task emb not found, start building task embeddings...')
        shape = build_emb_bank(testset, emb_model, 'task', testset_emb_path)
        logging.info(f'task emb built successfully. a numpy matrix of shape {shape} is stored at {testset_emb_path}')
    testset_emb_mat = np.load(testset_emb_path).astype('float32')
    logging.info(f"loaded testset task embeddings of shape: `{testset_emb_mat.shape}`")
    if args.method == 'adaretrieve':
        adaretrieve(args.results_path)
    elif args.method == 'decompretrieve':
        gen_model = OpenAIChatModel(model_name=args.gen_model)
        k_example = args.k_example
        es = ExampleSelecter(example_bank_path=args.example_bank_path, emb_field=args.example_emb_field, emb_model=emb_model, examplebank_emb_path=args.examplebank_emb_path)
        if not os.path.exists(es.examplebank_emb_path):
            logging.info('examplebank_emb not found.')
            es.build_example_emb_pool()
        decomposer = Decomposer(gen_model)
        reranker = Reranker(gen_model)
        library=args.library
        k_subtask_api = args.k_subtask_api
        # decompretrieve_batch(args.subtask_emb_path, args.example_subtask_output_path, args.api_recommendation_output_path)
        decompretrieve_pipeline(args.api_recommendation_output_path)
    else:
        raise ValueError(f"Unsupported method: {args.method}")
    
    os.chdir(cwd)
    