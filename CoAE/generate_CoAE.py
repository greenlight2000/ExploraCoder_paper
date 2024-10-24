import contextlib
import shutil
import logging
import re
import subprocess
import os
import sys
import subprocess
import random
from pathlib import Path
import pandas as pd
import numpy as np
from utils import setup_log
from models import BaseGenModel, OpenAIChatModel
from typing import List, Dict
from cot_generator import APIExplorationGenerator, APIExplorationDebugger
from collections import defaultdict
from utils import read_df
import argparse

def run_subprocess_in_conda_env(executable, conda_env_path):
    """
    Take an executable python code string and run it in a conda environment.
    Return the return code and the output, error of the code execution.
    """
    py_cmd = f'{conda_env_path}/bin/python'
    process = subprocess.Popen([py_cmd, '-c', executable], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    output = output.decode().strip()
    return process.returncode, output, error
def check_observation(solution, timeout=3, task_id='', subtask_id='', candidate_id=''):
    """
    call the `check_observation` function in another conda environment, to get the result of one code snippet execution.
    return a dict `result`: {'task_id': task_id, 'solution_id': solution_id, 'status': status, 'output': output, 'exception': exception}
    - status: "passed" or "timed out" or "failed" or "system error"
    - output: the output of the code execution, such as print() output
    - exception: the exception message of the target solution code, or (when the status is "system error") the excption message of the proxy code in this function.
    """
    proxy_code = """\
import sys
import os
sys.path.append({utils_dir})
sys.path.append({execution_dir})
from execution import check_observation
program = r'''
{program}
'''
result = check_observation('t{task_id}','s{subtask_id}'+'_'+'c{candidate_id}',program,{timeout})
print(result)
"""
    cur_dir = Path(__file__).absolute().parent
    utils_dir = Path(cur_dir).parent.parent
    execution_dir = Path(cur_dir).parent / 'evaluate'
    executable = proxy_code.format(
        utils_dir=f'"{utils_dir}"',
        execution_dir=f'"{execution_dir}"',
        program=solution,
        task_id=task_id,
        subtask_id=subtask_id,
        candidate_id=candidate_id,
        timeout=timeout)
    conda_env_path = 'replace/to/your/library/conda/env'
    subprocess_code, output, error = run_subprocess_in_conda_env(executable, conda_env_path)

    if subprocess_code != 0:
        result = {'task_id': task_id, 'solution_id': subtask_id, 'status': 'system error', 'output': output, 'excpetion': f'unexpected sys error in conda subprocess: {error.decode()}'}
        logging.error(f"unexpected sys error in conda subprocess: {error.decode()}")
    else:
        try:
            result = eval(output)
        except Exception as e:
            print("output:",output)
            result = {'task_id': task_id, 'solution_id': subtask_id, 'status': 'system error', 'output': output, 'exception': f'unexpected sys error when parsing output from subprocess: {e}'}
            logging.error(f"unexpected sys error when parsing output from subprocess: {e}")
    return result


save_code2file_tmp = """\
# [SUBTASK]: {subtask}
# [PLAYGROUND CODE]:
{code}
# [OBSERVAION]:
'''
# [STATUS]: {status}
# [OUTPUT]:
{output}
# [EXCEPTION]
{exception}
'''\
"""
def generate_exploration(testset, external_data_source, ar_results, ar_field, code_generator, code_debugger, code_parser, output_dir, show_api_format, dodebug, max_retry=5):
    with create_evaldir(evaldir=output_dir, external_data_path=external_data_source):
        error_index = []
        for i,row in testset.iterrows():
            task_id_num = re.search(r'\d+', row['task_id']).group()
            logging.info("========================================================================")
            logging.info(f"Processing task {i}, task_id={row['task_id']}")
            logging.info("========================================================================")
            task = row['task']
            prompt = row['prompt']

            ar_result = ar_results.query(f"task==@task").reset_index(drop=True).loc[0]
            subtasks = ar_result['decomposed_subtasks']

            cot_explore_path: List[dict] = []
            for subtask_id, subtask in enumerate(subtasks, start=1):
                logging.info('----------------------------------------------------------------')
                logging.info(f'start explorating on t{task_id_num}_st{subtask_id}')
                logging.info('----------------------------------------------------------------')
                # 1. Node Expansion
                logging.info(f'generating cot codes for subtask {subtask_id}')
                subtask_relevant_api_df = pd.DataFrame(ar_result[ar_field][subtask_id-1])
                retry_cnt = 0
                while True:
                    if retry_cnt > max_retry:
                        logging.error(f"Retry times exceed {max_retry}. Returning original codes.")
                        code_snippets = ['']*code_generator.n
                        error_index.append(task_id_num)
                        break
                    try:
                        code_snippets,_ = code_generator.generate(prompt, cot_explore_path, subtask_id, subtask, subtask_relevant_api_df, show_api_format)
                    except Exception as e:
                        logging.error(f"Generator Error when generating code for task_id {task_id_num}, subtask {subtask_id}.{subtask}: {e}. Regenerating...")
                        retry_cnt += 1
                        continue
                    if all(code_generator.check_validity(code_snippet) for code_snippet in code_snippets):
                        logging.info(f"Validity check passed: task {row['task_id']} subtask {subtask_id}. Sanitizing...")
                        code_snippets = [code_generator.sanitize(code_snippet) for code_snippet in code_snippets]
                        break
                    logging.info(f"Validity check failed: task {row['task_id']} subtask {subtask_id}. Regenerating...")
                    retry_cnt += 1

                if code_snippets[0] == '':
                    continue    

                # 2. Reward Calculation: 
                logging.info(f'start exploring the candidates cot code for subtask {subtask_id}')
                cot_tree_records:Dict[List[dict]] = defaultdict(list)
                for candidate_id, code in enumerate(code_snippets, start=1):
                    result = check_observation(code, 15) 
                    logging.info(f"t{task_id_num}_st{subtask_id}_c{candidate_id}: {result['status']}\n{result['exception']}")
                    candidate_cur_explore_state = {
                        'subtask_id': subtask_id,
                        'candidate_id': candidate_id, 
                        'refine_id': '0', 
                        'subtask': subtask,
                        'code': code, 
                        'used_api_df': parse_used_apis(ar_result, code),
                        'status': result['status'],
                        'output': result['output'],
                        'exception': result['exception']
                    }
                    cot_tree_records[subtask_id].append(candidate_cur_explore_state)

                    cot_codes_dir = Path(output_dir) / task_id_num
                    if not os.path.exists(str(cot_codes_dir)):
                        os.makedirs(str(cot_codes_dir), exist_ok=True)
                    code2file = save_code2file_tmp.format(
                        subtask = subtask,
                        code=code,
                        status=result['status'],
                        output=result['output'],
                        exception=result['exception']
                    )
                    open(str(cot_codes_dir / Path(f'{get_node_path_snapshot(task_id_num, [*cot_explore_path, candidate_cur_explore_state])}.py')), 'w').write(code2file)
                # 3. Node Selection: 
                cur_explore_state:dict = None 
                for candidate_cur_explore_state in cot_tree_records[subtask_id]:
                    if candidate_cur_explore_state['status']=='passed':
                        cur_explore_state = candidate_cur_explore_state
                        break
                debug_records = []
                if dodebug and cur_explore_state is None:
                    logging.info(f'all candidates failed to execute, start debuging candidates')
                    for candidate in cot_tree_records[subtask_id]:
                        candidate_id = candidate['candidate_id']
                        err_code = candidate['code']
                        err_message = candidate['exception']

                        retry_cnt =0
                        while True:
                            if retry_cnt > max_retry:
                                logging.error(f"Retry times exceed {max_retry}. Returning 'error' codes.")
                                debug_code = 'failed generation'
                                error_index.append(task_id_num)
                                break
                            try:
                                debug_codes,_ = code_debugger.debug(err_code, err_message, subtask_relevant_api_df, show_api_format)
                            except Exception as e:
                                logging.error(f"Debugger Error when debugging code for task_id {task_id_num}, subtask {subtask_id}.{subtask}, candidate_id {candidate_id}: {e}. Regenerating...")
                                retry_cnt += 1
                                continue
                            if all(code_debugger.check_validity(debug_code) for debug_code in debug_codes):
                                logging.info(f"Validity check passed: task {row['task_id']} subtask {subtask_id}. Sanitizing...")
                                debug_code = code_debugger.sanitize(debug_codes[0])
                                break
                            logging.info(f"Validity check failed: task {row['task_id']} subtask {subtask_id}. Regenerating...")
                            retry_cnt += 1

                        result = check_observation_proxy(debug_code, 15)
                        logging.info(f"t{task_id_num}_st{subtask_id}_c{candidate_id}_debug: {result['status']}\n{result['exception']}")
                        candidate_cur_explore_state = {
                            'subtask_id': subtask_id,
                            'candidate_id': candidate_id, 
                            'refine_id': '1', 
                            'subtask': subtask,
                            'code': debug_code, 
                            'used_api_df': parse_used_apis(ar_result, debug_code),
                            'status': result['status'],
                            'output': result['output'],
                            'exception': result['exception']
                        }
                        debug_records.append(candidate_cur_explore_state)
                        code2file = save_code2file_tmp.format(
                            subtask = subtask,
                            code=debug_code,
                            status=result['status'],
                            output=result['output'],
                            exception=result['exception']
                        )
                        open(str(cot_codes_dir / Path(f'{get_node_path_snapshot(task_id_num, [*cot_explore_path, candidate_cur_explore_state])}.py')), 'w').write(code2file)
                        if candidate_cur_explore_state['status']=='passed':
                            cur_explore_state = candidate_cur_explore_state
                            break
                    cot_tree_records[subtask_id].extend(debug_records)
                
                if cur_explore_state is None:
                    logging.info(f'all candidates failed to execute, put the first failed one into the explore path anyway.')
                    cur_explore_state = cot_tree_records[subtask_id][random.randint(0, len(cot_tree_records[subtask_id])-1)] if len(debug_records)==0 else debug_records[random.randint(0, len(debug_records)-1)]
                cot_explore_path.append(cur_explore_state)
                explore_trajectory_str = "\n".join([f"({thought['subtask_id']}. subtask: {subtask}, candidate: {thought['candidate_id']}, debug: {thought['refine_id']=='1'}, status: {thought['status']}, err: {thought['exception']})" for thought in cot_explore_path])
                logging.info(f'current explore thoughts:\n{explore_trajectory_str}')

                cot_path_filenames = [get_node_path_snapshot(task_id_num, cot_explore_path[:i+1]) for i in range(len(cot_explore_path))]
                with open(cot_codes_dir / Path("selected_cot_path.txt"), 'w') as f:
                    for fn in cot_path_filenames:
                        f.write(fn+'\n')
                logging.info(f'--------finished exploration on t{task_id_num}_st{subtask_id}--------')
            logging.info(f"==========finished inference on task {i}, task_id={row['task_id']}==========")

        if len(error_index)>0:
            logging.error(f"error happened on task_id: {error_index}")
            print(f"error happened on task_id: {error_index}")
        else:
            logging.info("successfully finished all tasks!")
            print("successfully  finished all tasks!")


def get_node_path_snapshot(task_id, cot_explore_path:List[dict])->str:
    """return a snapshot of the node path, which is a unique identifier of the path.
    """
    cur_explore_state = cot_explore_path[-1]
    path_snapshot = f't{task_id}'+''.join([f"_s{thought['subtask_id']}c{thought['candidate_id']}o{int(thought['status']=='passed')}" for thought in cot_explore_path])
    if cur_explore_state['refine_id']=='1':
        path_snapshot += '_debug'
    return path_snapshot

import pandas as pd
import re
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


import contextlib
import shutil
@contextlib.contextmanager
def create_evaldir(evaldir, external_data_path):# sandbox for executing experimental codes
    cwd = os.getcwd()
    if evaldir is None:
        evaldir = cwd
    os.makedirs(evaldir, exist_ok=True)
    os.chdir(evaldir)
    logging.info(f"Creating and changing directory to evaldir: {os.getcwd()}.")
    try:
        if external_data_path is not None:
            folder_name = os.path.basename(external_data_path)
            dst_dir = os.path.join(evaldir, folder_name)
            shutil.copytree(external_data_path, dst_dir, dirs_exist_ok=True)
            logging.info(f"Copying external data from {external_data_path} to {dst_dir}")
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)
        logging.info(f"Changing directory back to {cwd}.")

def get_args():
    parser = argparse.ArgumentParser(description="Argument parser for the script")
    parser.add_argument('--testset_path', type=str)
    parser.add_argument('--external_data_source', type=str)
    parser.add_argument('--ar_path', type=str)
    parser.add_argument('--ar_field', type=str)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo-0125')
    parser.add_argument('--explore_n', type=int, default=5)
    parser.add_argument('--explore_temperature', type=float, default=0.8)
    parser.add_argument('--explore_top_p', type=float, default=0.95)
    parser.add_argument('--debug_n', type=int, default=1)
    parser.add_argument('--debug_temperature', type=float, default=0)
    parser.add_argument('--debug_top_p', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--dodebug', action='store_true', default=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    cwd = os.getcwd()
    os.chdir(Path(__file__).parent)
    args = get_args()

    external_data_source = str(Path(args.external_data_source).expanduser().resolve(strict=True))
    output_dir = str(Path(args.output_dir).expanduser().resolve())
    if os.path.exists(output_dir):
        input(f"Output directory already exists:{args.output_dir}.\nPress Enter to clear the directory and continue...")
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    os.makedirs(output_dir, exist_ok=True)

    setup_log(Path(output_dir)/Path('CoAE_generation.log'), console=False)
    logging.info(f"Args:\n{args}")

    testset = read_df(args.testset_path)
    ar_results = pd.read_json(args.ar_path, orient='records', lines=True)
    gen_model = OpenAIChatModel(model_name=args.model_name)
    code_generator = APIExplorationGenerator(gen_model, n=args.explore_n, temperature=args.explore_temperature, top_p=args.explore_top_p)
    code_debugger = APIExplorationDebugger(gen_model, n=args.debug_n, temperature=args.debug_temperature, top_p=args.debug_top_p)
    code_parser = CodeGenParser()

    generate_exploration(
        testset, 
        external_data_source, 
        ar_results, 
        args.ar_field, 
        code_generator, 
        code_debugger, 
        code_parser, 
        output_dir, 
        'basics', 
        args.dodebug
    )
    os.chdir(cwd)