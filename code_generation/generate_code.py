from typing import List, Tuple
import pandas as pd
import re
import os
from pathlib import Path
import sys
import logging
from utils import num_tokens, read_df, setup_log
from utils import CodeGenParser
from models import OpenAIChatModel
import argparse   

def codegen_with_exploration_n_apis(coae_dir, ar_results, ar_field, k_cg_api, results_path, show_api_format):
    error_index = []
    for i,row in testset.iterrows():
        task_id = row['task_id']
        task_id_num = re.findall(r'\d+', task_id)[0]
        logging.info("========================================================================")
        logging.info(f"Processing task {i}, task_id={task_id}")
        logging.info("========================================================================")
        task = row['task']
        prompt = row['prompt']
        ar_result = ar_results.query(f"task==@task").reset_index(drop=True).loc[0]

        task_coae_dir = Path(coae_dir) / Path(task_id_num)
        
        cot_path_filenames:List[str] = []
        if os.path.exists(task_coae_dir / Path("selected_cot_path.txt")):
            with open(task_coae_dir / Path("selected_cot_path.txt"), 'r') as f:
                cot_path_filenames = f.readlines()
                cot_path_filenames = [fn.strip() for fn in cot_path_filenames]
        else:
            cot_path_filenames = parse_coae_path(task_coae_dir)
            with open(task_coae_dir / Path("selected_cot_path.txt"), 'w') as f:
                for fn in cot_path_filenames:
                    f.write(fn+'\n')

        cot_path_results: List[dict] = []
        for cot_path_filename in cot_path_filenames:
            subtask, code, status, output, exception = get_node_info(task_coae_dir / Path(cot_path_filename+'.py'))
            used_api_df = parse_used_apis(ar_result, code)
            cot_path_results.append({'subtask': subtask, 'code': code, 'used_api_df': used_api_df, 'status': status, 'output': output, 'exception': exception})

        rec_api_df = pd.DataFrame(ar_result[ar_field]).loc[:k_cg_api]

        retry_cnt = 0
        max_retry = 5
        while True:
            if retry_cnt > max_retry:
                logging.error(f"Retry times exceed {max_retry}. Returning original codes.")
                code_snippets = ['']*code_generator.n
                error_index.append(task_id_num)
                break
            try:
                code_snippets,_ = code_generator.generate_with_cot_api(task, cot_path_results, rec_api_df, prompt, show_api_format)
            except Exception as e:
                logging.error(f"Generator Error when generating code for task_id {task_id_num}: {e}. Regenerating...")
                retry_cnt += 1
                continue
            if any(code_generator.check_validity(code_snippet) for code_snippet in code_snippets):
                logging.info(f"Validity check passed: task {row['task_id']}. Sanitizing...")
                code_snippets = [code_generator.sanitize(code_snippet) for code_snippet in code_snippets]
                break
            logging.info(f"Validity check failed: task {row['task_id']}. Regenerating...")
            retry_cnt += 1

        if code_snippets[0] == '':
            results = pd.DataFrame([[task_id, "", "", "", "", ""]],columns=['task_id', 'task', 'prompt', "test", "canonical_solution", 'generated_solutions'])
            results.to_json(results_path, orient='records', lines=True, mode='a')
            continue

        results = pd.DataFrame([[
            task_id,
            task,
            prompt,
            row['test'],
            row['canonical_solution'],
            code_snippets,
            [api['api_path'] for i,api in rec_api_df.iterrows()]
        ]],columns=['task_id', 'task', 'prompt', 'test', "canonical_solution", 'generated_solutions','recommended_apis'])
        results.to_json(results_path, orient='records', lines=True, mode='a')
        logging.info(f"------------------------------finished task {i}------------------------------")
    if len(error_index) != 0:
        logging.warning(f"Error in processing tasks index={error_index}. please check the log file for details and rerun the corresponding tasks")
    else:
        logging.info(f"Finished inference.")

def parse_coae_path(task_coae_dir):
    cot_path = []
    filenames = os.listdir(str(task_coae_dir))
    pattern = re.compile(r's\d+c\d+o\d+')
    sequences_count = {fn: len(pattern.findall(fn)) for fn in filenames}
    max_sequence_length = max(sequences_count.values())
    longest_sequence_files = [fn for fn, count in sequences_count.items() if count == max_sequence_length]
    longest_sequence_files.sort()
    o1_files = [fn for fn in longest_sequence_files if fn.endswith('o1.py') or fn.endswith('o1_debug.py')]
    if o1_files:
        selected_file = o1_files[0]
    else:
        debug_files = [fn for fn in longest_sequence_files if fn.endswith('_debug.py')]
        if debug_files:
            selected_file = debug_files[0]
        else:
            selected_file = longest_sequence_files[0]
    cot_path.append(selected_file)
    while True:
        selected_file = re.sub(r'(_s\d+c\d+o\d+(_debug)?)\.py$', '.py', selected_file)
        if not re.search(r's\d+c\d+o\d+', selected_file):
            break
        else:
            if selected_file.replace('.py','_debug.py') in filenames:
                selected_file = selected_file.replace('.py','_debug.py')
            assert selected_file in filenames, f"File {selected_file} not found in filenames."
            cot_path.append(selected_file)
    return cot_path[::-1]

def get_node_info(file_path):
    contents = open(file_path).read()
    subtask_n_code, observation = contents.split("\n# [OBSERVAION]:\n")[0], contents.split("\n# [OBSERVAION]:\n")[1].replace("'''", "")
    subtask, code = subtask_n_code.split("# [PLAYGROUND CODE]:")[0].replace("# [SUBTASK]:","").strip(), subtask_n_code.split("# [PLAYGROUND CODE]:")[1].strip()
    status_n_output, exception = observation.split("# [EXCEPTION]\n")[0], observation.split("# [EXCEPTION]\n")[1].strip()
    status, output = status_n_output.split("# [OUTPUT]:\n")[0].replace("# [STATUS]:","").strip(), status_n_output.split("# [OUTPUT]:\n")[1].strip()
    return subtask, code, status, output, exception

def parse_used_apis(ar_result, code):
    df_list =[pd.DataFrame(subtask_apis) for subtask_apis in ar_result['retrieved_apis_per_subtask']]
    total_rec_apis = pd.concat(df_list)
    total_rec_apis = total_rec_apis.drop_duplicates(subset=['api_path']).reset_index(drop=True)

    api_search_space = total_rec_apis
    used_api_idxs = []
    for api_idx,row in api_search_space.iterrows():
        api_name_pattern = rf"[^a-zA-Z]{row['api_name']}\("
        match = re.search(api_name_pattern, code)
        if match:
            used_api_idxs.append((api_idx,match.start()))
        elif type(row['api_functional_name'])==str and row['api_functional_name']!='':
            fun_name_pattern = '\.' + row['api_functional_name'].split('`')[0] + '\('
            match = re.search(fun_name_pattern, code)
            if match:
                used_api_idxs.append((api_idx,match.start()))
    used_api_idxs = [e[0] for e in sorted(used_api_idxs, key=lambda x:x[1])]
    used_apis = api_search_space.loc[used_api_idxs].reset_index(drop=True)
    return used_apis


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse arguments for the script.")

    parser.add_argument('--log_path', type=str)
    parser.add_argument('--testset_path', type=str)
    parser.add_argument('--ar_path', type=str)
    parser.add_argument('--ar_field', type=str)
    parser.add_argument('--k_cg_api', type=int, default=20)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--cot_dir', type=str)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo-0125')
    parser.add_argument('--show_api_format', type=str, default='basics')
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--n', type=int)

    args = parser.parse_args()

    return args

from code_generator import FinalSolutionGenerator
if __name__ == '__main__':
    cwd = os.getcwd()
    os.chdir(Path(__file__).parent)
    
    args = parse_arguments()
    setup_log(args.log_path, console=False)

    testset = read_df(args.testset_path)
    ar_results = pd.read_df(args.ar_path)
    model = OpenAIChatModel(model_name=args.model_name)
    code_generator = FinalSolutionGenerator(model, n=args.n, temperature=args.temperature, top_p=args.top_p)
    code_parser = CodeGenParser()
    codegen_with_exploration_n_apis(args.cot_dir, ar_results, args.ar_field, args.k_cg_api, args.results_path, args.show_api_format)
    os.chdir(cwd)