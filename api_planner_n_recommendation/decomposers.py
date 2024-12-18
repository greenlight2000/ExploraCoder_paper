from models import BaseGenModel
from typing import List
import pandas as pd

DOCOMPOSER_PROMPT = """\
I will give you a task that needs to use several Torchdata APIs to implement it. You need to break down the task into several subtasks.
[Torchdata library]
{Torchdata_description}

{few_shot_examples}
[Task]
{task}
[Subtasks]
"""

Torchdata_description = """\
Torchdata is a library of common modular data loading primitives for constructing flexible data pipelines. It introduces composable Iterable-style and Map-style building blocks called DataPipes, which work well with PyTorch's DataLoader and have functionalities for loading, parsing, caching, transforming, and filtering datasets. 
DataPipes can be composed together into datasets and support execution in various settings and execution backends using DataLoader2. 
The library aims to make data loading components more flexible and reusable by providing a new DataLoader2 and modularizing features of the original DataLoader into DataPipes. 
DataPipes are a renaming and repurposing of the PyTorch Dataset for composed usage, allowing for easy chaining of transformations to reproduce sophisticated data pipelines. 
DataLoader2 is a light-weight DataLoader that decouples data-manipulation functionalities from torch.utils.data.DataLoader and offers additional features such as checkpointing/snapshotting and switching backend services for high-performant operations.\
"""
class Decomposer():
    def __init__(self, model:BaseGenModel):
        self.model = model

    def decompose(self, user_req:str, examples:pd.DataFrame)->List[str]:
        example_str = ""
        for n, (_,item) in enumerate(examples.iterrows(),1):
            subtasks_str = "\n".join(f"{cnt}. {subtask}" for cnt, subtask in enumerate(item['subtasks'], start=1)) + "\n"
            example_str += f"[Task]\n{item['task']}\n[Subtasks]\n{subtasks_str}\n\n"
        prompt = DOCOMPOSER_PROMPT.format(Torchdata_description=Torchdata_description, few_shot_examples=example_str, task=user_req)
        retry = 0
        max_retry = 3
        while True:
            try:
                res = self.model.generate(prompt)
                print(res)
                subtasks = self.parse_list(res)
                if type(subtasks) == list and len(subtasks)>0:
                    break
                else:
                    raise Exception(f"subtasks not parsed into string list, subtasks: {subtasks}")
            except Exception as e:
                retry += 1
                if retry >= max_retry:
                    raise e
        return subtasks
    def parse_list(self, number_list_str):
        lines = number_list_str.splitlines()
        parsed_list = [line.split('. ', 1)[1] for line in lines if '. ' in line]
        return parsed_list