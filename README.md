This repo contains core algorithm implementations of ExploraCoder and the [Torchdata-Manual](./torchdata_manual.json) benchmarks we constructed for evaluating ExploraCoder and its baselines' performance on multi-API invocations. Some case studies for representative approaches on solving a example task (same in paper's Appendix) are also provided at [example.md](./example.md).

On Torchdata-Manual and Torchdata-Github benchmarks, the entries represent:
- task requirement (entry 'task')
- full programming problem. including task requirement and code context (entry 'prompt') 
- canonical solution for the problem (entry 'canonical_solution')
- test csaes for problem (entry 'test')

In the data files, we currently provide partial data samples. Full data will be released after paper being accepted. 
Other Implementation details can be found in our paper.
