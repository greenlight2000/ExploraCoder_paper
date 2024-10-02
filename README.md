This folder contains the two benchmarks we constructed for evaluating ExploraCoder and its baselines' performance on multi-API invocations.
 (1) Torchdata-Github
 (2) Torchdata-Manual

On each dataset, we provide:
- task requirement (entry 'task')
- full programming problem. including task requirement and code context (entry 'prompt') 
- canonical solution for the problem (entry 'canonical_solution')
- test csaes for problem (entry 'test')

In the data files, we only provide 5 samples per benchmark. Full data along with code implementation will be released after paper being accepted. Implementation details can be found in our paper.