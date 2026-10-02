# eXCore

**A hardware-aware CPU quantization search engine and decentralized benchmark registry.**

Most model-optimization subnets make miners compete on one expensive GPU. That keeps
participation thin: few developers can afford or source the hardware just to compete.

eXCore moves the whole evaluation loop to **CPU and system RAM**. Anyone with a laptop can mine.

## How it works

Miners submit a lightweight YAML **manifest**: a layer-by-layer map of GGUF quantization formats
(`Q3_K`, `Q4_K`, `Q5_K`, `Q8_0`, ...) for a 27B-parameter model with 273 searchable units. Instead of one
blanket quantization that makes the model dumb, miners search for the sweet spot: compress the
easy layers hard to save RAM, keep the sensitive layers precise to preserve accuracy.

```yaml
schema: excore/manifest@1
track: CPU-01
name: attn-q8-mlp-q4
default: Q5_K
rules:
  - { match: "L*.mlp",     format: Q4_K }
  - { match: "L*.attn.*",  format: Q8_0 }
  - { match: "lm_head",    format: Q6_K }
```

A validator bot picks up each pull request, reads the manifest **as data** (it never runs PR code),
builds the model, and benchmarks it on CPU through llama.cpp (AVX2 / NEON). The result is placed on a
**4-D Pareto frontier**:

| objective | how it is measured |
|---|---|
| accuracy drift | Reference-Partition KL-divergence against the unquantized model |
| decode speed | tokens/s |
| prefill speed | tokens/s on a 4K prompt |
| peak RAM | high-water mark of the benchmark process |

If a recipe pushes the frontier forward (faster, or leaner, without destroying accuracy) and passes
the gates, the bot auto-merges it into an evolving registry of hyper-optimized CPU models and assigns
a reward tier from `core:S` to `core:Ultra`.

Recipes the repository owner merges by hand are evaluated and recorded too, so they count the same
way, and the owner can approve a recipe the frontier rejected with a label (the gates and holdout can
never be overridden).

## Quick start

**Miners**

```bash
pip install -e .
excore check manifests/my-recipe.yaml        # legal? roughly how big?
# open a PR that adds exactly one new file: manifests/my-recipe.yaml
```

See [docs/miner_guide.md](docs/miner_guide.md).

**Validator operators** follow [evaluator/runbook.md](evaluator/runbook.md).

**Developers**

```bash
pip install -e ".[dev]"
pytest
```

## Documentation

| | |
|---|---|
| [docs/miner_guide.md](docs/miner_guide.md) | write, check and submit recipes |
| [docs/precision_space.md](docs/precision_space.md) | legal formats, aliases, quantizers |
| [docs/core_frontier.md](docs/core_frontier.md) | the scoring system, gates and reward tiers, with a worked example |
| [evaluator/runbook.md](evaluator/runbook.md) | hosting a validator node |

## Repository layout

```
eXCore/
├── .github/workflows/    pr_evaluator.yml (the bot) and ci.yml (tests)
├── configs/              hpc_cpu.yaml (every evaluation constraint), sources.lock.json (pinned hashes)
├── data/corpus/          hash-locked public text for accuracy measurement
├── docs/                 miner guide, precision space, scoring
├── evaluator/            bot_engine.py, runner.py, sandbox.py, github.py, runbook.md
├── excore/               the Python package
│   ├── manifest.py       strict YAML parser: recipe -> per-unit assignments
│   ├── precision.py      legal GGUF formats and who can produce them
│   ├── build.py          deterministic compiler: manifest + source -> candidate GGUF
│   ├── validate.py       audit: byte-level checks before anything runs
│   ├── gguf.py           dependency-free GGUF reader/writer
│   ├── model/            the 273-unit model map (configurable per track)
│   ├── quantizers/       plugin contract; runtime_gguf (llama.cpp), local_rtn (numpy RTN)
│   ├── eval/             accuracy (RP-KL), performance, holdout, llama.cpp backend
│   ├── frontier/         core_frontier (4-D Pareto), gates, rewards, plots
│   ├── sources.py        hash-locked sources
│   ├── screen.py         pre-submission size estimate
│   └── cli.py            `excore check | build | audit | sources | plots | reference`
├── experiments/          V0 baseline and V1 MLP-focused starting recipes
├── manifests/            where miners' submissions land (<name>.yaml)
├── results/core_spectrum/  spectrum.json (the registry) and rendered plots
├── scripts/              setup_bench.sh (pinned llama.cpp build), download_models.sh
└── tests/
```

## Design principles

- **Untrusted code never runs.** Miner PRs are one YAML file, parsed as data by a strict parser.
- **Reproducible.** The same manifest and source always produce the same bytes; the audit checks them.
- **Noise cannot buy a place.** Candidates are scored conservatively against incumbents.
- **Hash-locked.** The source weights, llama.cpp commit and public corpus are pinned.
- **Everything configurable lives in one file:** `configs/hpc_cpu.yaml`.

## Status

The pipeline is complete and unit-tested end to end with stand-ins for llama.cpp. Before opening to
miners, an operator must pin the real sources, populate the corpus, build the references and
calibrate the bounds. See the "Known limitations" section of the runbook.

## License

MIT. See [LICENSE](LICENSE).
