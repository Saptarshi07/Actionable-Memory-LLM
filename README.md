# Actionable Memory of LLMs in Cooperation Games

This repository contains code and data for experiments on large language model (LLM) behavior in repeated two-player cooperation games.

The main collection script presents an LLM with every possible history of a repeated game up to a fixed horizon, asks it to choose between actions `L` and `R`, and saves the resulting history--action data. These data can be used to study which features of past interaction histories best explain LLM choices.

## Overview

For a horizon of \(R=4\), there are

\[
4^4 = 256
\]

possible histories of joint actions. The script queries each model a fixed number of times for every history.

For example, with:

```python
TARGET_COPIES_PER_HISTORY = 6
```

the final dataset contains:

```text
256 histories × 6 responses per history = 1536 records
```

for each model and treatment.

The script supports:

- OpenAI-compatible models, including GPT-4o and GPT-5.6-Sol;
- Anthropic Claude models accessed through the Harvard API;
- Meta Llama models accessed through the Harvard Bedrock API proxy.

## Repository structure

```text
.
├── run-experiments.py                 # Main data-collection script
├── data/                           # History--action datasets
│   ├── claude-sonnet-4-6/
│   ├── gpt-4o-temp0/
│   ├── gpt-5.6-sol/
│   └── llama-4-maverick/
├── analysis.py                        # script for conditional entropy analysis
├── README.md
└── .gitignore
```

Each model and treatment has a separate JSON data file. For example:

```text
data/llama-4-maverick/interchange-LR/data_llm_H4_T00_norev_words.json
```

## Installation

Install the Python dependencies:

```bash
pip install openai requests numpy pandas matplotlib
```

The collection script otherwise uses standard Python libraries, including `os`, `json`, `random`, `re`, `time`, and `collections`.

## API key setup

To run run-experiments.py you would need a API-KEY of your own,
In addition you will have to replace all api_url variables with your relevent api_url.
(ours was an api-url at Harvard.)


## Configuring an experiment

The main configuration block is near the top of the collection script:

```python
TEMPERATURE = 0
TARGET_COPIES_PER_HISTORY = 6
HORIZON = 4
ACTIONS = ["L", "R"]

REVERSE = False
TREATMENT = "words"

aa, ab, ba, bb = 3, 0, 5, 1

MAPPING_ID = 1
```

### Main parameters

| Variable | Meaning |
|---|---|
| `HORIZON` | Number of past rounds shown to the model. |
| `TARGET_COPIES_PER_HISTORY` | Number of independent responses collected for each possible history. |
| `TEMPERATURE` | Sampling temperature for supported APIs. |
| `REVERSE` | If `False`, histories are presented from oldest to most recent; if `True`, they are presented from most recent to oldest. |
| `TREATMENT` | History formatting option. Current options include `"words"` and `"random"`. |
| `aa, ab, ba, bb` | Payoffs for outcomes `(L,L)`, `(L,R)`, `(R,L)`, and `(R,R)`. |
| `MAPPING_ID` | Action-label mapping presented in the prompt. |

The available label mappings are:

```python
LABEL_MAPPINGS = [
    {"L": "L", "R": "R"},  # identity mapping
    {"L": "R", "R": "L"},  # interchange L and R
]
```

## Experimental treatments

### Baseline treatment

```python
REVERSE = False
MAPPING_ID = 0
aa, ab, ba, bb = 3, 0, 5, 1
```

### Interchange-LR treatment

```python
MAPPING_ID = 1
```

This switches the meanings of `L` and `R` in the prompt. Histories and responses are nevertheless stored in canonical `L/R` notation, allowing direct comparison with baseline data.

### Reverse-history treatment

```python
REVERSE = True
```

This presents the most recent round first rather than last.

### Alternation treatment

For example:

```python
aa, ab, ba, bb = 3, 0, 8, 1
```

Under these payoffs, alternation between asymmetric outcomes gives players a higher average payoff than sustained mutual cooperation.

### Donation-game treatment

For example:

```python
aa, ab, ba, bb = 2, -1, 3, 0
```

This corresponds to a donation game with benefit \(b=3\) and cost \(c=1\).

## Selecting models

The collection script loops over models in:

```python
for MODEL_NAME in [
    "gpt-4o",
    "gpt-5.6-sol",
    "us.anthropic.claude-sonnet-4-6",
    "us.meta.llama4-maverick-17b-instruct-v1:0",
]:
```

To run only one model, replace the list with, for example:

```python
for MODEL_NAME in ["gpt-4o"]:
```

Make sure that each model is assigned a distinct output path.

## Running data collection

Run:

```bash
python run-experiments.py
```

The script:

1. Generates all possible histories of length `HORIZON`;
2. Loads any previously collected data;
3. Counts the number of saved responses for each history;
4. Queries only histories that still need responses;
5. Saves the entire dataset after every successful API response;
6. Verifies that every history appears exactly `TARGET_COPIES_PER_HISTORY` times.

## Resuming interrupted runs

The data-collection process is resumable.

If collection stops midway through a run, all successfully completed responses have already been saved. When the script is run again, it reloads the existing JSON file and queries only the histories that still have fewer than the requested number of copies.

For example, if the target is three copies per history and a run stops after 300 records, rerunning the script completes the missing copies until every one of the 256 histories appears exactly three times.

## Data format

Each JSON record has the form:

```json
{
  "history": [
    ["L", "R"],
    ["R", "R"],
    ["L", "L"],
    ["R", "L"]
  ],
  "action_next": "L",
  "mapping_id": 1
}
```

where:

- `history` is the past interaction history in chronological order;
- the first action in each pair is the focal model's action;
- the second action in each pair is the co-player's action;
- `action_next` is the model's next action;
- `mapping_id` records the label mapping used in the prompt.

### Canonical storage convention

Histories and actions are stored in canonical `L/R` notation, including in the `interchange-LR` treatment.

For example, under the swapped-label prompt, the model may literally return `R` to select the action that is canonically labeled `L`. The script converts this response back before saving:

```json
{
  "action_next": "L"
}
```

This convention makes the datasets directly comparable across prompt treatments.

## Output validation

The final dataset size is:

```python
TARGET_COPIES_PER_HISTORY * (4 ** HORIZON)
```

For the default configuration:

```text
6 × 4^4 = 1536 records
```

The script verifies that every possible history appears exactly the requested number of times.


## Analysis

The analysis script measures how well different summaries of the recent game history explain an LLM's next action.

For a horizon of \(R=4\), each past round has four possible joint outcomes:

```text
LL, LR, RL, RR
```

There are fifteen possible ways to group these four outcomes into nonempty categories, called **partitions**. For example:

```text
{{LL}, {LR, RL, RR}}
```

isolates mutual `L` play from all other outcomes, while:

```text
{{LL}, {LR, RL}, {RR}}
```

is a counting partition: it distinguishes zero, one, and two `L` actions in a round.

For each one-round partition \(P\), the analysis constructs its product partition \(P^k\) over the previous \(k\) rounds. It then computes the empirical conditional entropy

$$
\mathrm{CE}_{P^k},
$$

which measures how unpredictable the model's next action remains after knowing only the partition block containing the recent history.

The benchmark is the full partition:

$$
P_0 =
\bigl\{
\{LL\},
\{LR\},
\{RL\},
\{RR\}
\bigr\},
$$

whose product \((P_0)^k\) retains the complete \(k\)-round history. For every non-full partition, the script reports:

$$
\mathrm{CE}_{P^k}
-
\mathrm{CE}_{(P_0)^k}.
$$

A value near zero means that the coarser partition explains the model's choices almost as well as the complete history. Larger values indicate that the partition discards distinctions that are important for predicting the model's action.

The script also reports the mean difference across lookback lengths:

$$
\frac{1}{R+1}
\sum_{k=0}^{R}
\left[
\mathrm{CE}_{P^k}
-
\mathrm{CE}_{(P_0)^k}
\right].
$$

Non-full partitions are ranked from lowest to highest mean difference. Thus, the top-ranked partition provides the most compact explanation of the data among the candidate partitions.

Run the analysis with:

```bash
python analysis.py \
  data/claude-sonnet-4-6/version-2/data_llm_H4_T00_norev_words.json
```

To also print LaTeX tables:

```bash
python analysis.py \
  data/claude-sonnet-4-6/version-2/data_llm_H4_T00_norev_words.json \
  --latex
```

To save results in JSON format:

```bash
python analysis.py \
  data/claude-sonnet-4-6/version-2/data_llm_H4_T00_norev_words.json \
  --output results.json
```


## Notes on Llama output

Llama is asked to return only `L` or `R`. The script uses a short generation length and retries invalid outputs. A record is written only after a valid action has been extracted.

If invalid Llama outputs occur repeatedly, inspect the raw API response and consider adjusting the prompt or output-generation settings.

## Reproducibility notes

- API models may change over time, even if their model names do not change.
- Provider-side settings may affect outputs despite deterministic sampling settings.
- The experiments use stateless API calls: each relevant history is included explicitly in the prompt.
- Results may differ for persistent model agents that retain interaction history through an active context window.


## Security

Never commit API keys, tokens, credentials, or private configuration files. If an API key is exposed in source code, a notebook, terminal output, screenshot, or Git history, revoke and replace it immediately.