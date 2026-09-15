# The Polymath Initiative

## Experiment #6 — Graph nodes → graph traversal

`task_eval_cleaned_revised/graph_memory.py` evaluates LoCoMo with a factual
memory graph and `meta-llama/Llama-3.2-3B-Instruct`.

For every conversation, the experiment:

1. extracts stated facts from each session as provenance-preserving graph edges;
2. grounds each question in named graph nodes;
3. breadth-first traverses both incoming and outgoing facts for up to two hops;
4. ranks the traversed facts and answers only from that compact evidence.

Run a small smoke evaluation after accepting the Llama model licence and
installing its inference packages (`pip install torch transformers`) with:

```powershell
python task_eval_cleaned_revised/graph_memory.py `
  --data locomo10.json `
  --conversation 0 `
  --max-questions 3 `
  --output graph_results.json
```

Use `--all-conversations` to run the complete LoCoMo file. The output records
the question seed entities and every retrieved fact, including session, date,
and dialogue provenance when it is supplied by the extractor.

The graph logic itself has no model dependency at import time and can be tested
with:

```powershell
python -m unittest discover -s task_eval_cleaned_revised/tests -v
```
