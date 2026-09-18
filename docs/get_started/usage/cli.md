---
title: "Command Line"
icon: lucide/terminal
---

# Command Line

This described the is the command line interface for `mteb`.

`mteb` is a toolkit for evaluating the quality of embedding models on various benchmarks. It supports the following commands:

- [`mteb run`](#running-models-on-tasks): Runs a model on a set of tasks
- [`mteb available_tasks`](#listing-available-tasks): Lists the available tasks within MTEB
- [`mteb available_benchmarks`](#listing-available-benchmarks): Lists the available benchmarks
- [`mteb create_meta`](#creating-model-metadata): Creates the metadata for a model card from a folder of results
- [`mteb leaderboard`](#running-the-leaderboard): Runs the MTEB leaderboard locally
- [`mteb mock-run`](#checking-model-implementations): Sanity checks a model implementation using mock tasks

In the following we outline some sample use cases, but if you want to learn more about the arguments for each command you can run:

```bash
mteb {command} --help
```

## Running Models on Tasks

To run a model on a set of tasks, use the `mteb run` command. For example:

```bash
mteb run -m sentence-transformers/average_word_embeddings_komninos \
         -t Banking77Classification EmotionClassification \
         --output-folder mteb_output
```

This will create a folder `mteb_output/{model_name}/{model_revision}` containing the results of the model on the specified tasks supplied as a json
file; `{task_name}.json`.

### Multi-GPU encoding (single node)

`--device` accepts a single device or a comma-separated list. A single value runs on
one device; a list enables **single-node multi-GPU encoding**, spreading the encode
work across the GPUs via SentenceTransformers' multi-process pool:

```bash
# single GPU
mteb run -m <model> -t NFCorpus --device 0

# all 8 GPUs on the node
mteb run -m <model> -t NFCorpus --device 0,1,2,3,4,5,6,7
```

!!! note "How it works"
    When a device list is given, each `encode` call forwards it to
    `SentenceTransformer.encode`, which **spawns a multi-process pool for that call**,
    shards the batch across the GPUs, and tears the pool down afterwards. No persistent
    pool is held — this keeps the change minimal. The spawn cost (a few tens of seconds,
    dominated by each worker importing the Python dependencies) is therefore paid **per
    call**; on a **networked filesystem** it is much larger, so installing the package on
    node-local storage (or a container image) makes it markedly faster.

!!! tip "Tuning the corpus chunk size for large retrieval corpora"
    For retrieval, MTEB encodes the corpus in chunks of `corpus_chunk_size` (default
    **50,000**), and each chunk is one multi-GPU `encode` call — i.e. **one pool spawn per
    chunk**. On a large corpus (millions of docs → hundreds of chunks) that per-call spawn
    can dominate the wall-clock. Increasing the chunk size amortises it (fewer, bigger
    calls), roughly toward `chunk ≈ len(corpus) / num_gpus` so each GPU gets ~one shard per
    call.

    The cap is **GPU memory**: brute-force search computes the full similarity block
    `n_queries × chunk` in fp32 on a single GPU, so keep
    `n_queries × chunk × 4 bytes` within the available memory (leaving headroom for the
    embeddings). Practically: `chunk ≈ min(len(corpus) / num_gpus, gpu_free_bytes × ~0.4 /
    (n_queries × 4))`, floored at the 50,000 default. This chunk size is a
    `SearchEncoderWrapper` parameter (Python API); it is **not** auto-tuned and is not yet
    exposed as a CLI flag. For very large corpora where the per-call spawn is the
    bottleneck, prefer the multi-node distributed evaluation (which shards the search too).


## Listing Available Tasks

To list the available tasks within MTEB, use the `mteb available-tasks` command. For example:

```bash
mteb available-tasks # list _all_ available tasks
```

You can also use the multiple arguments for filtering:
```
mteb available-tasks --task-types Retrieval --languages eng # list all English (eng) retrieval tasks
```

## Listing Available Benchmarks

To list the available benchmarks within MTEB:

```bash
mteb available-benchmarks # list all available benchmarks
```


## Creating Model Metadata

Once a model is run you can create the metadata for a model card from a folder of results, use the `mteb create-meta` command. For example:

```bash
mteb create-meta --results-folder mteb_output/sentence-transformers__average_word_embeddings_komninos/{revision} \
                 --output-path model_card.md
```

This will create a model card at `model_card.md` containing the metadata for the model on MTEB within the YAML frontmatter. This will make the model
discoverable on the MTEB leaderboard.

## Running the Leaderboard

To run the MTEB leaderboard locally, use the `mteb leaderboard` command. For example:

```bash
mteb leaderboard
```

You can specify a custom cache path and other options:

```bash
mteb leaderboard --cache-path results --port 8080 --share
```

Available options:
- `--cache-path PATH`: Custom path for model results cache
- `--host HOST`: Host to run the server on (default: 0.0.0.0)
- `--port PORT`: Port to run the server on (default: 7860)
- `--share`: Create a public URL for the leaderboard
- `--rebuild`: Force rebuild from full results repository, bypassing cached JSON

For more details on running the leaderboard, see the [leaderboard documentation](leaderboard.md).

## Checking Model Implementations

To sanity check a model implementation using a set of example test tasks, use the `mteb mock-run` command. This command evaluates the model on a set of tasks covering various modalities and task types, while not requiring any external datasets, making it a fast way to test that the implementation works as intended before running larger benchmarks or when implementing a model.

For example:

```bash
mteb mock-run -m sentence-transformers/average_word_embeddings_komninos
```

This will run the model on all compatible mock tasks, print a Markdown summary table to stdout, and save the markdown results to `mteb_mock_run_results.md`.

Available options:
- `-m, --model MODEL`: The model to use. Prioritizes the model implementation from MTEB's model registry, or defaults to loading via `sentence-transformers`.
- `--model-revision REVISION`: Revision of the model to load.
- `--device DEVICE`: Device(s) for computation. A single value (e.g. `cpu`, `0`, `cuda:0`) uses one device; a comma-separated list (e.g. `0,1,2,3`) enables single-node multi-GPU encoding.
- `-v, --verbosity VERBOSITY`: Verbosity level (0 to 4, default: 2).

The same checks are available from Python using [`mteb.mock_run`](../../contributing/adding_a_model.md#local-model-verification-using-mock-tasks), which returns the per-task status instead of writing a file.
