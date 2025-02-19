
# Qualitative example
Given an example task in our benchmark Torchdata-Manual. (The problem style aligns with previous library-oriented benchmark such as Pandas-Eval and Torchdata-Eval)

The task requirement is 
```
Load text data from files under a given directory, process the text data into paragraphs, create mini-batches of paragraphs, and shuffle the batches. The processed data should be cached and held in memory.
```
Given the context code below, LLM are asked to implement a function.
```
from torchdata.datapipes.iter import *
from typing import List, Tuple
import torch

def process_paragraphs_map_fn(paragraph: Tuple[str, str]) -> str:
    file_name = paragraph[0]
    paragraph_content = paragraph[1]
    return paragraph_content

def build_text_data_pipe(
        root: str,  # Directory containing text files
        batch_size: int,  # Batch size
    ):
    """
    Load text data from files under a given directory, process the text data into paragraphs, create mini-batches of paragraphs, and shuffle the batches. The processed data should be cached and held in memory.
    """
```
---
Following are the inference processes of varying-styled approaches (including [Standard RAG](#docprompting-failed), [Iterative Debugging](#capir--self-repair-failed), [Agentic](#react-failed), [ExploraCoder](#exploracoder-success)) trying to solve this example problem.
## Docprompting (failed)
The retrieved APIs are:
```
torchdata.datapipes.iter.ParagraphAggregator
torchdata.datapipes.map.Batcher
torchdata.datapipes.iter.Batcher
torchdata.datapipes.iter.OnDiskCacheHolder
torchdata.datapipes.iter.InBatchShuffler
torchdata.datapipes.iter.BucketBatcher
torchdata.datapipes.iter.JsonParser
torchdata.datapipes.iter.Shuffler
torchdata.datapipes.map.InMemoryCacheHolder
torchdata.datapipes.iter.FullSync
torchdata.datapipes.iter.CSVParser
torchdata.datapipes.iter.Prefetcher
torchdata.datapipes.iter.OnlineReader
torchdata.datapipes.iter.CSVDictParser
torchdata.datapipes.iter.ParquetDataFrameLoader
torchdata.datapipes.map.Shuffler
torchdata.datapipes.iter.RandomSplitter
torchdata.datapipes.map.Concater
torchdata.datapipes.iter.LineReader
torchdata.datapipes.iter.FileLister
```
One generated candidate seems promising but fails to execute:
```
# Define a function to process paragraphs
def process_paragraphs_map_fn(paragraph: Tuple[str, str]) -> str:
    file_name = paragraph[0]
    paragraph_content = paragraph[1]
    return paragraph_content
# Build the text data pipeline
def build_text_data_pipe(
    root: str="./torchdata-programming-tasks/task_82", # Directory containing text files
    batch_size: int=16, # Batch size
):
    file_list_dp = FileLister(root=root, masks=["*.txt"])
    line_reader_dp = LineReader(file_list_dp) # wrong API usage
    paragraph_aggregator_dp = ParagraphAggregator(line_reader_dp)
    batcher_dp = Batcher(paragraph_aggregator_dp, batch_size=batch_size)
    shuffler_dp = InBatchShuffler(batcher_dp)
    cached_dp = InMemoryCacheHolder(shuffler_dp)
    return cached_dp
```
### Analysis
In this case, GPT-3.5-turbo + naive RAG successfully recalled most relevant APIs to generator. However, it
hallucinate on the usage of `FileLister` and `LineReader` APIs, since `FileLister` outputs the
filename, while ‘LineReader‘ expects file stream as input to read each line from the stream. The end-to-end style generation overlook the intermediate step of using `FileOpener` or self-implemented the file-reading action to turn outputs from `FileLister` into stream. Such
mistake hinders the successful generation of correct code solutions for multi-API problems.

## CAPIR + Self-Repair (failed)
The preactively planned & recommended APIs from CAPIR's recommendation module are:
```
torchdata.datapipes.iter.IterableWrapper
torchdata.datapipes.iter.FileLister
torchdata.datapipes.iter.IoPathFileLister
torchdata.datapipes.iter.FSSpecFileLister
torchdata.datapipes.iter.FileOpener
torchdata.datapipes.iter.IoPathFileOpener
torchdata.datapipes.iter.LineReader
torchdata.datapipes.iter.ParagraphAggregator
torchdata.datapipes.iter.Batcher
torchdata.datapipes.iter.InBatchShuffler
torchdata.datapipes.map.Shuffler
torchdata.datapipes.map.InMemoryCacheHolder
torchdata.datapipes.iter.InMemoryCacheHolder
torchdata.datapipes.iter.OnDiskCacheHolder
torchdata.datapipes.iter.Prefetcher
torchdata.datapipes.map.Concater
torchdata.datapipes.iter.Concater
torchdata.datapipes.iter.UnBatcher
torchdata.datapipes.iter.Shuffler
torchdata.datapipes.iter.FullSync
```
The initial generated code from CAPIR was:
```
from torchdata.datapipes.iter import *
from typing import List, Tuple
import torch
def process_paragraphs_map_fn(paragraph: Tuple[str, str]) -> str:
    file_name = paragraph[0]
    paragraph_content = paragraph[1]
    return paragraph_content
def build_text_data_pipe(
    root: str, # Directory containing text files
    batch_size, # Batch size
):
    data_pipe = FileLister(root) >> IoPathFileLister() >> FileOpener() >> LineReader()
    data_pipe = data_pipe.map(process_paragraphs_map_fn) >> Batcher(batch_size=batch_size) >> InMemoryCacheHolder()
    data_pipe = data_pipe >> Shuffler()
    return data_pipe

[execution result]
Status: failed
Error message:
TypeError: __init__() missing 1 required positional argument: ’root’
Error traceback at:
- line 18 in build_text_data_pipe: data_pipe = FileLister(root) >> IoPathFileLister() >> FileOpener() >> LineReader()
```
The initial code failed, and Self-Repair begins to iteratively debug on this code:

Iteration 1
```
[feedback]
The error message indicates that the ‘FileLister‘ class in the ‘build_text_data_pipe‘ function is missing the
required positional argument ‘root‘. The ‘FileLister‘ class expects the root directory containing text
files as an argument. To resolve this error, make sure to provide the ‘root‘ argument when creating the
‘FileLister‘ instance in the ‘build_text_data_pipe‘ function.
[code]
from torchdata.datapipes.iter import *
from typing import List, Tuple
import torch
def process_paragraphs_map_fn(paragraph: Tuple[str, str]) -> str:
    file_name = paragraph[0]
    paragraph_content = paragraph[1]
    return paragraph_content
def build_text_data_pipe(
    root: str="./torchdata-programming-tasks/task_82", # Directory containing text files
    batch_size: int=16, # Batch size
):
    """
    Load text data from files under a given directory, process the text data into paragraphs, create
    mini-batches of paragraphs, and shuffle the batches. The processed data should be cached and held in
    memory.
    """
    data_pipe = FileLister(root=root) >> IoPathFileLister() >> FileOpener() >> LineReader()
    data_pipe = data_pipe.map(process_paragraphs_map_fn) >> Batcher(batch_size=batch_size) >>
    InMemoryCacheHolder()
    data_pipe = data_pipe >> Shuffler()
    return data_pipe
    list_dp = list(build_text_data_pipe(root="./torchdata-programming-tasks/task_82", batch_size=4))
[execution result]
Status: failed
Error message:
TypeError: __init__() missing 1 required positional argument: ’root’
Error traceback at:
- line 23 in <module>: list_dp = list(build_text_data_pipe(root="./torchdata-programming-tasks/task_82",
batch_size=4))
- line 18 in build_text_data_pipe: data_pipe = FileLister(root=root) >> IoPathFileLister() >>
FileOpener() >> LineReader()
```
Iteration 2
```
[feedback]
The error message indicates that the ‘FileLister‘ class in the ‘build_text_data_pipe‘ function is missing the
required positional argument ‘root‘. The ‘FileLister‘ class expects the root directory containing text
files as an argument. To resolve this error, make sure to provide the ‘root‘ argument when creating the
‘FileLister‘ instance in the ‘build_text_data_pipe‘ function.
[code]
from torchdata.datapipes.iter import *
from typing import List, Tuple
import torch
def process_paragraphs_map_fn(paragraph: Tuple[str, str]) -> str:
    file_name = paragraph[0]
    paragraph_content = paragraph[1]
    return paragraph_content
def build_text_data_pipe(
    root: str="./torchdata-programming-tasks/task_82", # Directory containing text files
    batch_size: int=16, # Batch size
):
    """
    Load text data from files under a given directory, process the text data into paragraphs, create
    mini-batches of paragraphs, and shuffle the batches. The processed data should be cached and held in
    memory.
    """
    data_pipe = FileLister(root=root) >> IoPathFileLister() >> FileOpener() >> LineReader()
    data_pipe = data_pipe.map(process_paragraphs_map_fn) >> Batcher(batch_size=batch_size) >>
    InMemoryCacheHolder()
    data_pipe = data_pipe >> Shuffler()
    return data_pipe
    list_dp = list(build_text_data_pipe(root="./torchdata-programming-tasks/task_82", batch_size=4))
[execution result]
Status: failed
Error message:
TypeError: __init__() missing 1 required positional argument: ’root’
Error traceback at:
- line 23 in <module>: list_dp = list(build_text_data_pipe(root="./torchdata-programming-tasks/task_82",
batch_size=4))
- line 18 in build_text_data_pipe: data_pipe = FileLister(root=root) >> IoPathFileLister() >>
FileOpener() >> LineReader()
```
We omit iteration 3-5 for simplicity. (They are too long and are basically futile debugging efforts.)

Iteration 6
```
[feedback]
The error message indicates that the ’root’ argument is missing when creating the FileLister data pipe. The FileLister API expects the root argument to be provided, but it is not being passed correctly in the build_text_data_pipe function. To resolve this issue, ensure that the ’root’ argument is correctly passed when creating the FileLister data pipe within the build_text_data_pipe function.
[code]
from torchdata.datapipes.iter import *
from typing import List, Tuple
import torch
def process_paragraphs_map_fn(paragraph: Tuple[str, str]) -> str:
    file_name = paragraph[0]
    paragraph_content = paragraph[1]
    return paragraph_content
def build_text_data_pipe(
    root: str = "./torchdata-programming-tasks/task_82", # Directory containing text files
    batch_size: int = 16, # Batch size
):
    """
    Load text data from files under a given directory, process the text data into paragraphs, create
    mini-batches of paragraphs, and shuffle the batches. The processed data should be cached and held in
    memory.
    """
    data_pipe = FileLister(root=[root]) >> IoPathFileLister() >> FileOpener() >> LineReader()
    data_pipe = data_pipe.map(process_paragraphs_map_fn) >> Batcher(batch_size=batch_size) >>
    InMemoryCacheHolder()
    data_pipe = data_pipe >> Shuffler()
    return data_pipe
list_dp = list(build_text_data_pipe(root="./torchdata-programming-tasks/task_82", batch_size=4))
[execution result]
Status: failed
Error message:
TypeError: __init__() missing 1 required positional argument: ’root’
Error traceback at:
- line 23 in <module>: list_dp = list(build_text_data_pipe(root="./torchdata-programming-tasks/task_82",
batch_size=4))
- line 18 in build_text_data_pipe: data_pipe = FileLister(root=[root]) >> IoPathFileLister() >>
FileOpener() >> LineReader()
```
### Analysis
First of all, CAPIR's reranking mechanism successfully recall more accurate APIs. However, in the initial code solution, the candidate severely misuses the Torchdata library. Specifically, it
incorrectly assumes that Torchdata APIs can be chained using ">>", and this incorrect API usage pattern propagates throughout the entire LLM-decoded API invocation sequence. Despite receiving clear error messages in each round of Self-Repair, the system fails to correct the code because its initial
solution deviates significantly from the canonical implementation.

## ReAct (failed)
ReAct reactively generate thoughts and actions to complete the task. We **additionally** instruct with trajectory example to guide the LLM to **generate partial code at each steps**. This is to examine **whether existing agentic methods can enforce the step-wise code construction feature** that we propose. The action space of ReAct Agent are:
1. Retrieve: Retrieve APIs with clearly described target functionality (such as a subtask in the coding task) from Torchdata API documentation and return the top-5 relevant API infomation.
2. Write: Write your temporary code snippet and execute it, the execution result will be returned. You don't need to impelement all requirement in the function at once, just write a temporary code snippet to test the partial subtask-wise functionality.
3. Finish: Write the complete code solution that solves the provided coding task.  

The inference-time trajectory of ReAct is shown below:

Reasoing step 1:
```
Thought: The task involves loading text data from files, processing into paragraphs, creating mini-batches, shuffling the batches, and caching the processed data in memory. Let's start by searching for Torchdata APIs related to file operations and text processing.
Action: Retrieve[[file operations and text processing APIs in Torchdata]]
Observation:
torchdata.datapipes.iter.ParagraphAggregator
torchdata.datapipes.iter.OnlineReader
torchdata.datapipes.iter.IoPathFileOpener
torchdata.datapipes.map.MapDataPipe
torchdata.datapipes.iter.JsonParser
```
Reasoing step 2:
```
Thought: Now that we have relevant APIs for file operations and text processing, we can start implementing the data pipeline. Let's first focus on reading files and aggregating lines into paragraphs.
Action: Write[[
\```python
def build_text_data_pipe(
        root: str="./torchdata-programming-tasks/task_82",  # Directory containing text files
        batch_size: int=16,  # Batch size
    ):
    # Read lines from files and aggregate into paragraphs
    file_list_dp = FileLister(root)
    file_list_dp = IterableWrapper(file_list_dp)
    file_opener_dp = IoPathFileOpener(file_list_dp)
    line_reader_dp = OnlineReader(file_opener_dp)
    paragraph_aggregator_dp = ParagraphAggregator(line_reader_dp, joiner=lines_to_paragraphs)
    
    # Test the pipeline
    for paragraph in paragraph_aggregator_dp:
        print(f"Paragraph: {paragraph[:50]}...")
        break
\```
]]
Observation:
failed
---------output---------
No output
---------exception---------
NameError: name 'FileLister' is not defined
Error traceback at:
 - line 17 in <module>: list_dp = list(build_text_data_pipe(root="./torchdata-programming-tasks/task_82", batch_size=4))
 - line 7 in build_text_data_pipe:     file_list_dp = FileLister(root)
```
Reasoing step 3:
```
Thought: The NameError indicates that the required Torchdata APIs are not imported. Let's import the necessary APIs and try running the code snippet again.
Action: Write[[
\```python
from torchdata.datapipes.iter import *
from typing import List, Tuple
import torch

def build_text_data_pipe(
        root: str="./torchdata-programming-tasks/task_82",  # Directory containing text files
        batch_size: int=16,  # Batch size
    ):
    # Read lines from files and aggregate into paragraphs
    file_list_dp = FileLister(root)
    file_list_dp = IterableWrapper(file_list_dp)
    file_opener_dp = IoPathFileOpener(file_list_dp)
    line_reader_dp = OnlineReader(file_opener_dp)
    paragraph_aggregator_dp = ParagraphAggregator(line_reader_dp, joiner=lines_to_paragraphs)
    
    # Test the pipeline
    for paragraph in paragraph_aggregator_dp:
        print(f"Paragraph: {paragraph[:50]}...")
        break
\```
]]
Observation:
failed
---------output---------
No output
---------exception---------
NameError: name 'lines_to_paragraphs' is not defined
Error traceback at:
 - line 21 in <module>: list_dp = list(build_text_data_pipe(root="./torchdata-programming-tasks/task_82", batch_size=4))
 - line 15 in build_text_data_pipe:     paragraph_aggregator_dp = ParagraphAggregator(line_reader_dp, joiner=lines_to_paragraphs)
```
Reasoing step 4:
```
Thought: The NameError indicates that the `lines_to_paragraphs` function is not defined. Let's define the function and rerun the code snippet.
Action: Write[[
\```python
from torchdata.datapipes.iter import *
from typing import List, Tuple
import torch

def lines_to_paragraphs(lines: List[str]) -> str:
    return ' '.join(lines)

def build_text_data_pipe(
        root: str="./torchdata-programming-tasks/task_82",  # Directory containing text files
        batch_size: int=16,  # Batch size
    ):
    # Read lines from files and aggregate into paragraphs
    file_list_dp = FileLister(root)
    file_list_dp = IterableWrapper(file_list_dp)
    file_opener_dp = IoPathFileOpener(file_list_dp)
    line_reader_dp = OnlineReader(file_opener_dp)
    paragraph_aggregator_dp = ParagraphAggregator(line_reader_dp, joiner=lines_to_paragraphs)
    
    # Test the pipeline
    for paragraph in paragraph_aggregator_dp:
        print(f"Paragraph: {paragraph[:50]}...")
        break
\```
]]
Observation:
failed
---------output---------
No output
---------exception---------
AttributeError: 'tuple' object has no attribute 'decode'
This exception is thrown by __iter__ of OnlineReaderIterDataPipe(skip_on_error=False, source_datapipe=IoPathFileOpenerIterDataPipe, timeout=None)
Error traceback at:
 - line 24 in <module>: list_dp = list(build_text_data_pipe(root="./torchdata-programming-tasks/task_82", batch_size=4))
 - line 21 in build_text_data_pipe:     for paragraph in paragraph_aggregator_dp:
 - line 112 in <genexpr>: return tuple(x.decode(encoding, errors) if x else '' for x in args)
```
Reasoing step 5:
```
Thought: The AttributeError indicates an issue with decoding the data. It seems the data is not in the expected format. Let's investigate further by checking the data being processed.
Action: Write[[
\```python
from torchdata.datapipes.iter import *
from typing import List, Tuple
import torch

def lines_to_paragraphs(lines: List[str]) -> str:
    return ' '.join(lines)

def build_text_data_pipe(
        root: str="./torchdata-programming-tasks/task_82",  # Directory containing text files
        batch_size: int=16,  # Batch size
    ):
    # Read lines from files and aggregate into paragraphs
    file_list_dp = FileLister(root)
    file_list_dp = IterableWrapper(file_list_dp)
    file_opener_dp = IoPathFileOpener(file_list_dp)
    line_reader_dp = OnlineReader(file_opener_dp)
    paragraph_aggregator_dp = ParagraphAggregator(line_reader_dp, joiner=lines_to_paragraphs)
    
    # Test the pipeline
    for paragraph in paragraph_aggregator_dp:
        print(f"Paragraph: {paragraph[:50]}...")
        print(f"Type of paragraph: {type(paragraph)}")
        break
\```
]]
Observation:
failed
---------output---------
No output
---------exception---------
AttributeError: 'tuple' object has no attribute 'decode'
This exception is thrown by __iter__ of OnlineReaderIterDataPipe(skip_on_error=False, source_datapipe=IoPathFileOpenerIterDataPipe, timeout=None)
Error traceback at:
 - line 25 in <module>: list_dp = list(build_text_data_pipe(root="./torchdata-programming-tasks/task_82", batch_size=4))
 - line 21 in build_text_data_pipe:     for paragraph in paragraph_aggregator_dp:
 - line 112 in <genexpr>: return tuple(x.decode(encoding, errors) if x else '' for x in args)
```
Reasoing step 6:
```
Thought: The AttributeError is related to decoding the data, indicating a mismatch in data format. The data being processed might not be in the expected format. Let's review the Torchdata APIs used for reading files and check the data format being processed.
Action: Retrieve[[data format handling APIs in Torchdata]]
Observation:
torchdata.datapipes.map.MapDataPipe
torchdata.datapipes.map.Batcher
torchdata.datapipes.iter.JsonParser
torchdata.datapipes.iter.IterDataPipe
torchdata.datapipes.iter.OnlineReader
```
Reasoing step 7:
```
Thought: The data format handling APIs in Torchdata do not directly address the decoding issue we are facing. We need to investigate the data format being processed and potentially handle the decoding manually if needed. Let's check the data format and make necessary adjustments.

Action: Write[[
\```python
from torchdata.datapipes.iter import *
from typing import List, Tuple
import torch

def lines_to_paragraphs(lines: List[str]) -> str:
    return ' '.join(lines)

def build_text_data_pipe(
        root: str="./torchdata-programming-tasks/task_82",  # Directory containing text files
        batch_size: int=16,  # Batch size
    ):
    # Read lines from files and aggregate into paragraphs
    file_list_dp = FileLister(root)
    file_list_dp = IterableWrapper(file_list_dp)
    file_opener_dp = IoPathFileOpener(file_list_dp)
    line_reader_dp = OnlineReader(file_opener_dp)
    paragraph_aggregator_dp = ParagraphAggregator(line_reader_dp, joiner=lines_to_paragraphs)
    
    # Test the pipeline
    for paragraph in paragraph_aggregator_dp:
        print(f"Paragraph: {paragraph[:50]}...")
        print(f"Type of paragraph: {type(paragraph)}")
        print(f"Content of paragraph: {paragraph}")
        break
\```
]]

Observation:
failed
---------output---------
No output
---------exception---------
AttributeError: 'tuple' object has no attribute 'decode'
This exception is thrown by __iter__ of OnlineReaderIterDataPipe(skip_on_error=False, source_datapipe=IoPathFileOpenerIterDataPipe, timeout=None)
Error traceback at:
 - line 26 in <module>: list_dp = list(build_text_data_pipe(root="./torchdata-programming-tasks/task_82", batch_size=4))
 - line 21 in build_text_data_pipe:     for paragraph in paragraph_aggregator_dp:
 - line 112 in <genexpr>: return tuple(x.decode(encoding, errors) if x else '' for x in args)
```
Reasoing step 8:
```
Thought: The AttributeError persists, indicating that the data format issue is not resolved. We need to handle the decoding of data manually to address this problem. Let's modify the code to handle data decoding explicitly.

Action: Write[[
\```python
from torchdata.datapipes.iter import *
from typing import List, Tuple
import torch

def lines_to_paragraphs(lines: List[str]) -> str:
    return ' '.join(lines)

def build_text_data_pipe(
        root: str="./torchdata-programming-tasks/task_82",  # Directory containing text files
        batch_size: int=16,  # Batch size
    ):
    # Read lines from files and aggregate into paragraphs
    file_list_dp = FileLister(root)
    file_list_dp = IterableWrapper(file_list_dp)
    file_opener_dp = IoPathFileOpener(file_list_dp)
    line_reader_dp = OnlineReader(file_opener_dp)
    paragraph_aggregator_dp = ParagraphAggregator(line_reader_dp, joiner=lines_to_paragraphs)
    
    # Manually handle decoding
    for file_name, content in paragraph_aggregator_dp:
        decoded_content = content.decode('utf-8') if isinstance(content, bytes) else content
        print(f"File: {file_name}, Decoded Content: {decoded_content[:50]}...")
        break
\```
]]

Observation:
failed
---------output---------
No output
---------exception---------
AttributeError: 'tuple' object has no attribute 'decode'
This exception is thrown by __iter__ of OnlineReaderIterDataPipe(skip_on_error=False, source_datapipe=IoPathFileOpenerIterDataPipe, timeout=None)
Error traceback at:
 - line 25 in <module>: list_dp = list(build_text_data_pipe(root="./torchdata-programming-tasks/task_82", batch_size=4))
 - line 21 in build_text_data_pipe:     for file_name, content in paragraph_aggregator_dp:
 - line 112 in <genexpr>: return tuple(x.decode(encoding, errors) if x else '' for x in args)
```
Reasoing step 9:
```
Thought: The AttributeError persists even after handling data decoding manually. This indicates that the data format being processed is not as expected. We need to further investigate the data format and potentially adjust the processing steps to handle the data correctly.

Action: At this point, it seems that the data format being processed is not as expected, leading to decoding issues. To address this, we need to carefully review the Torchdata APIs used for file reading and data processing. Let's retrieve relevant APIs for data format handling and adjust our processing steps accordingly.

Observation: Invalid Action. Valid Actions are Retrieve[[<topic>]] Write[[<code>]] and Finish[[<solution>]].
```
Reasoing step 10:
```
Thought: The current approach of manually handling data decoding is not resolving the issue. We need to reconsider the data processing steps and potentially adjust the pipeline to handle the data format correctly. Let's retrieve relevant APIs for data format handling and make necessary adjustments to the processing steps.

Action: Retrieve[[data format handling APIs in Torchdata]]

Observation:
torchdata.datapipes.map.MapDataPipe
torchdata.datapipes.map.Batcher
torchdata.datapipes.iter.JsonParser
torchdata.datapipes.iter.IterDataPipe
torchdata.datapipes.iter.OnlineReader
```
Reasoning step 11:
```
Thought: The data format handling APIs in Torchdata do not directly address the decoding issue we are facing. We need to carefully review the data being processed and adjust our processing steps accordingly. Let's retrieve relevant APIs for data processing and make necessary adjustments to handle the data format correctly.

Action: Write[[
\```python
from torchdata.datapipes.iter import *
from typing import List, Tuple
import torch

def lines_to_paragraphs(lines: List[str]) -> str:
    return ' '.join(lines)

def build_text_data_pipe(
        root: str="./torchdata-programming-tasks/task_82",  # Directory containing text files
        batch_size: int=16,  # Batch size
    ):
    # Read lines from files and aggregate into paragraphs
    file_list_dp = FileLister(root)
    file_list_dp = IterableWrapper(file_list_dp)
    file_opener_dp = IoPathFileOpener(file_list_dp)
    line_reader_dp = OnlineReader(file_opener_dp)
    paragraph_aggregator_dp = ParagraphAggregator(line_reader_dp, joiner=lines_to_paragraphs)
    
    # Manually handle decoding and processing
    for file_name, content in paragraph_aggregator_dp:
        decoded_content = content.decode('utf-8') if isinstance(content, bytes) else content
        paragraphs = decoded_content.split('\n\n')  # Assuming paragraphs are separated by double newlines
        for paragraph in paragraphs:
            print(f"File: {file_name}, Paragraph: {paragraph[:50]}...")
        break
\```
]]

Observation:
failed
---------output---------
No output
---------exception---------
AttributeError: 'tuple' object has no attribute 'decode'
This exception is thrown by __iter__ of OnlineReaderIterDataPipe(skip_on_error=False, source_datapipe=IoPathFileOpenerIterDataPipe, timeout=None)
Error traceback at:
 - line 27 in <module>: list_dp = list(build_text_data_pipe(root="./torchdata-programming-tasks/task_82", batch_size=4))
 - line 21 in build_text_data_pipe:     for file_name, content in paragraph_aggregator_dp:
 - line 112 in <genexpr>: return tuple(x.decode(encoding, errors) if x else '' for x in args)
```
Reasoning step 12 omitted for another looped retrieval reasoning

Reasoning step 13: 
```
Thought: The data format handling APIs in Torchdata do not directly address the decoding issue we are facing. We need to carefully review the data being processed and adjust our decoding and processing steps accordingly. Let's retrieve relevant APIs for data format handling and make necessary adjustments to handle the data format correctly.

Action: Write[[
\```python
from torchdata.datapipes.iter import *
from typing import List, Tuple
import torch

def lines_to_paragraphs(lines: List[str]) -> str:
    return ' '.join(lines)

def build_text_data_pipe(
        root: str="./torchdata-programming-tasks/task_82",  # Directory containing text files
        batch_size: int=16,  # Batch size
    ):
    # Read lines from files and aggregate into paragraphs
    file_list_dp = FileLister(root)
    file_list_dp = IterableWrapper(file_list_dp)
    file_opener_dp = IoPathFileOpener(file_list_dp)
    line_reader_dp = OnlineReader(file_opener_dp)
    paragraph_aggregator_dp = ParagraphAggregator(line_reader_dp, joiner=lines_to_paragraphs)
    
    # Manually handle decoding and processing
    for file_name, content in paragraph_aggregator_dp:
        if isinstance(content, bytes):
            decoded_content = content.decode('utf-8')
        else:
            decoded_content = content
        paragraphs = decoded_content.split('\n\n')  # Assuming paragraphs are separated by double newlines
        for paragraph in paragraphs:
            print(f"File: {file_name}, Paragraph: {paragraph[:50]}...")
        break
\```
]]

Observation:
failed
---------output---------
No output
---------exception---------
AttributeError: 'tuple' object has no attribute 'decode'
This exception is thrown by __iter__ of OnlineReaderIterDataPipe(skip_on_error=False, source_datapipe=IoPathFileOpenerIterDataPipe, timeout=None)
Error traceback at:
 - line 30 in <module>: list_dp = list(build_text_data_pipe(root="./torchdata-programming-tasks/task_82", batch_size=4))
 - line 21 in build_text_data_pipe:     for file_name, content in paragraph_aggregator_dp:
 - line 112 in <genexpr>: return tuple(x.decode(encoding, errors) if x else '' for x in args)
```
Reasoning step 14-16 omitted for ineffective labor in debugging the error

Reasoning step 17
```
Thought: The decoding issue persists, indicating that the data format issue is not resolved. We need to further investigate the data format being processed and adjust our decoding and processing steps accordingly. Let's retrieve relevant APIs for data format handling and make necessary adjustments to handle the data format correctly.

Action: Finish[[
\```python
from torchdata.datapipes.iter import *
from typing import List, Tuple
import torch

def lines_to_paragraphs(lines: List[str]) -> str:
    return ' '.join(lines)

def build_text_data_pipe(
        root: str="./torchdata-programming-tasks/task_82",  # Directory containing text files
        batch_size: int=16,  # Batch size
    ):
    # Read lines from files and aggregate into paragraphs
    file_list_dp = FileLister(root)
    file_list_dp = IterableWrapper(file_list_dp)
    file_opener_dp = IoPathFileOpener(file_list_dp)
    line_reader_dp = OnlineReader(file_opener_dp)
    paragraph_aggregator_dp = ParagraphAggregator(line_reader_dp, joiner=lines_to_paragraphs)
    
    # Manually handle decoding and processing
    for file_name, content in paragraph_aggregator_dp:
        if isinstance(content, bytes):
            decoded_content = content.decode('utf-8')
        else:
            decoded_content = content
        paragraphs = decoded_content.split('\n\n')  # Assuming paragraphs are separated by double newlines
        for paragraph in paragraphs:
            print(f"File: {file_name}, Paragraph: {paragraph[:50]}...")
        break
\```
]]

Observation: Solution is INCORRECT
```
### Analysis:
At step 1-3, ReAct attempted to implement a partial functionality of “read files and aggregate lines”, with a trajectory of Retrieve-Implement-Debug. Although this is an temptation for planning step-wise code construction, we can observe such reactive planning style lack the systematic perspective and prior knowledge for the reasonable dependencies among subtask, resulting in an overly complex subtask that involve multiple Torchdata operations—read file(inherently involving FileLister and FileOpener two Torchdata APIs) and aggregate lines(missing a prior step of read file into lines). 
As a result, the outcome from step 1-3 is a complicated buggy code that (1) misuse the OnlineReader (2) hallucinated a joiner parameters for ParagraphAggregator.

The limitation of reactive planning lies in its **bug-driven nature**. In step 4 ReAct  tried to solve the bug “NameError: name 'lines_to_paragraphs' is not defined”, which derived from the hallucinated parameter, by introducing another hallucination: it implemented a lines_to_paragraphs method itself. This directly leads to a series of ineffective and meaningless debugging in step 4-8.

Another issue of reactive agents are its **uncontrollable stepping**. ReAct fell into looped action of retrieving “data format handling APIs in Torchdata” from Torchdata library in Step 10 and step 12, which increase the computational costs.

In the end, ReAct produce an incorrect solution. This show that **reactive planning is not enough** and **agentic workflow is ineffective/unstable for the step-wise code construction**.


## ExploraCoder (success)
ExploraCoder decomposed the example task into 6 subtask, and let LLM spontaneously generate exploratory subtask-level code and print the valueble information from API usage to gain insight.
We demonstrate the initial plans in planning module and the selected CoAE path here to show the exploration process of ExploraCoder.

Broad Plannings:
```
1. List all text files in the given directory
2. Open each text file and read the text data
3. Process the text data into paragraphs
4. Create mini-batches of paragraphs using torchdata APIs
5. Shuffle the batches using shuffle API
6. Cache and hold the processed data in memory
```
Subtask 1 on CoAE
```
[subtask]
1. List all text files in the given directory
[exploratory code]
import torchdata
from torchdata.datapipes.iter import FileLister
root = "./torchdata-programming-tasks/task_82"
file_lister = FileLister(root=root)
for file_path in file_lister:
    rint(file_path)
[observation]
Status: Executable
Output:
./torchdata-programming-tasks/task_82/file_0.txt
...omit 3 lines...
./torchdata-programming-tasks/task_82/file_4.txt
```
Subtask 2 on CoAE
```
[subtask]
2. Open each text file and read the text data
[exploratory code]
import torchdata
from torchdata.datapipes.iter import FileLister, IoPathFileOpener, LineReader
root = "./torchdata-programming-tasks/task_82"
file_lister = FileLister(root=root)
file_opener = IoPathFileOpener(source_datapipe=file_lister)
line_reader = LineReader(source_datapipe=file_opener)
for file_name, line in line_reader:
    print(f"File: {file_name}")
    print(f"Content: {line}")
[observation]
Status: Executable
Output:
File: ./torchdata-programming-tasks/task_82/file_0.txt
Content: This is line 0 of file 0
File: ./torchdata-programming-tasks/task_82/file_0.txt
...omit 995 lines...
File: ./torchdata-programming-tasks/task_82/file_4.txt
Content: This is line 99 of file 4
```
Subtask 3 on CoAE
```
[subtask]
3. Process the text data into paragraphs
[exploratory code]
import torchdata
from torchdata.datapipes.iter import FileLister, IoPathFileOpener, LineReader, ParagraphAggregator
from typing import Tuple
root = "./torchdata-programming-tasks/task_82"
file_lister = FileLister(root=root)
file_opener = IoPathFileOpener(source_datapipe=file_lister)
line_reader = LineReader(source_datapipe=file_opener)
paragraph_aggregator = ParagraphAggregator(source_datapipe=line_reader)
for file_name, paragraph in paragraph_aggregator:
    print(f"File: {file_name}")
    print(f"Paragraph: {paragraph}")
[observation]
Status: Executable
Output:
File: ./torchdata-programming-tasks/task_82/file_0.txt
Paragraph: This is line 0 of file 0
This is line 1 of file 0
...omit 500 lines...
This is line 98 of file 4
```
Subtask 4 on CoAE
```
[subtask]
4. Create mini-batches of paragraphs using torchdata APIs
[exploratory code]
import torchdata
from torchdata.datapipes.iter import FileLister, IoPathFileOpener, LineReader, ParagraphAggregator, Batcher
from typing import Tuple
root = "./torchdata-programming-tasks/task_82"
file_lister = FileLister(root=root)
file_opener = IoPathFileOpener(source_datapipe=file_lister)
line_reader = LineReader(source_datapipe=file_opener)
paragraph_aggregator = ParagraphAggregator(source_datapipe=line_reader)
batcher = Batcher(datapipe=paragraph_aggregator, batch_size=16)
for batch in batcher:
    print(batch)
[observation]
Status: Executable
Output:
[(’./torchdata-programming-tasks/task_82/file_0.tx...is line 98 of file 4\nThis is line 99 of file 4’)]
```
Subtask 5 on CoAE
```
[subtask]
5. Shuffle the batches using shuffle API
[exploratory code]
import torchdata
from torchdata.datapipes.iter import FileLister, IoPathFileOpener, LineReader, ParagraphAggregator, Batcher,
InBatchShuffler
root = "./torchdata-programming-tasks/task_82"
file_lister = FileLister(root=root)
file_opener = IoPathFileOpener(source_datapipe=file_lister)
line_reader = LineReader(source_datapipe=file_opener)
paragraph_aggregator = ParagraphAggregator(source_datapipe=line_reader)
batcher = Batcher(datapipe=paragraph_aggregator, batch_size=16)
shuffler = InBatchShuffler(datapipe=batcher)
for shuffled_batch in shuffler:
    print(shuffled_batch)
[observation]
Status: Executable
Output:
[(’./torchdata-programming-tasks/task_82/file_4.tx...is line 98 of file 2\nThis is line 99 of file 2’)]
```
Subtask 6 on CoAE
```
[subtask]
6. Cache and hold the processed data in memory
[exploratory code]
import torchdata
from torchdata.datapipes.iter import FileLister, IoPathFileOpener, LineReader, ParagraphAggregator, Batcher,
InMemoryCacheHolder
from typing import Tuple
root = "./torchdata-programming-tasks/task_82"
file_lister = FileLister(root=root)
file_opener = IoPathFileOpener(source_datapipe=file_lister)
line_reader = LineReader(source_datapipe=file_opener)
paragraph_aggregator = ParagraphAggregator(source_datapipe=line_reader)
batcher = Batcher(datapipe=paragraph_aggregator, batch_size=16)
cache_holder = InMemoryCacheHolder(source_dp=batcher)
for data in cache_holder:
    print(data)
[observation]
Status: Executable
Output:
[(’./torchdata-programming-tasks/task_82/file_0.tx...is line 98 of file 4\nThis is line 99 of file 4’)]
```

Eventually, based on the CoAE trace, the refined recommendation set of APIs are:
```
torchdata.datapipes.iter.FileLister
torchdata.datapipes.iter.IoPathFileOpener
torchdata.datapipes.iter.LineReader
torchdata.datapipes.iter.ParagraphAggregator
torchdata.datapipes.iter.Batcher
torchdata.datapipes.iter.Shuffler
torchdata.datapipes.iter.InMemoryCacheHolder
torchdata.datapipes.iter.IoPathFileLister
torchdata.datapipes.iter.FSSpecFileLister
torchdata.datapipes.iter.FileOpener
torchdata.datapipes.iter.InBatchShuffler
torchdata.datapipes.iter.OnDiskCacheHolder
torchdata.datapipes.map.Concater
torchdata.datapipes.iter.Concater
```
ExploraCoder use the refined APIs and the CoAE trace to enhance the solution plannings, generate the correct final solution:
```
from torchdata.datapipes.iter import *
from typing import List, Tuple
import torch
def build_text_data_pipe(
    root: str="./torchdata-programming-tasks/task_82", # Directory containing text files
    batch_size: int=16, # Batch size
):
    file_lister = FileLister(root=root)
    file_opener = IoPathFileOpener(source_datapipe=file_lister)
    line_reader = LineReader(source_datapipe=file_opener)
    paragraph_aggregator = ParagraphAggregator(source_datapipe=line_reader)
    batcher = Batcher(datapipe=paragraph_aggregator, batch_size=batch_size)
    shuffler = InBatchShuffler(datapipe=batcher)
    cache_holder = InMemoryCacheHolder(source_dp=shuffler)
    return cache_holder
```
### Analysis
We can find that ExploraCoder preactively plans a series of systematic subtasks. Then during our novel **CoAE**, ExploraCoder incrementally reuse APIs invocations from prior subtask, and experiments with new API invocations and gain new API knowledge from environment. **Such exploratory API knowledge will further compensate for the possible biased/under-specified preactive plans** (which we denote as "**exploratory planning**" in our Appendix A.1"). Eventually, ExploraCoder manage to enforce a "**step-wise code construction**" and find a successful API exploration trace, helping it to generate correct final solution.
