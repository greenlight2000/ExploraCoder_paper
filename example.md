
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
Following are inference process of three different approaches trying to solve this example problem.
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
The recommended APIs are:
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
One initial generated code from CAPIR was:
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

## ExploraCoder (success)
ExploraCoder decomposed the example task into 6 subtask, and let LLM spontaneously generate exploratory subtask-level code and print the valueble information from API usage to gain insight.
We demonstrate the selected CoAE path here to show the exploration process of ExploraCoder.

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
ExploraCoder use the refined APIs and the CoAE trace as CoT instruction, generate the correct final solution:
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
We can find that during CoAE, ExploraCoder increamentally reuse APIs invocations from prior subtask, and
experiments with new API invocations and observe their behaviour. Eventually, ExploraCoder manage to
find a successful API exploration trace, helping it to generate correct final solution.
