# Research Canvas Design

> Status: Working design / MVP specification  
> Purpose: Define an AI-native scientific workspace built around **Obsidian Canvas + Codex (or another local AI agent)** for research analysis, with SEEG as the initial use case.

---

# 1. Core Idea

The goal is not merely to use AI to write analysis code.

The goal is to create a shared workspace where:

- the researcher plans analyses visually;
- datasets, analyses, parameters, results, figures, decisions, warnings, and interpretations are represented as connected objects;
- an AI agent can read the same workspace;
- the agent can execute analyses;
- results can automatically appear back in the Canvas;
- important scientific reasoning and methodological decisions remain traceable;
- analysis history is reproducible without cluttering the main research view.

In this system:

- **Obsidian Canvas** = visual research state / shared research workspace
- **Codex / AI Agent** = planner + executor + critic
- **Python / MATLAB / R** = analysis execution
- **Git** = code history
- **run metadata** = execution history
- **Canvas** = scientific decision history

The Canvas should represent the **current scientific state of the project**, not every low-level code modification.

---

# 2. Do We Need Obsidian MCP?

## MVP: No

An Obsidian MCP server is **not required** for the first version.

The simplest architecture is:

```text
Researcher
    ↓
Obsidian
    ↓
.canvas / .md files
    ↕
Codex
    ↓
Python / MATLAB / R
    ↓
results / figures
    ↓
Codex updates .canvas
```

Codex only needs filesystem access to the project directory containing the Obsidian Vault.

Because `.canvas` files are JSON-based files, Codex can:

- read nodes;
- read edges;
- inspect labels;
- add nodes;
- change node content;
- replace a placeholder text node with a file/image node;
- preserve node position and ID;
- add reasoning summaries and results.

## MCP may become useful later

An Obsidian MCP integration may be helpful if we later want structured access to:

- Obsidian search;
- backlinks;
- tags;
- note metadata;
- plugin APIs;
- active note / active Canvas state;
- Obsidian commands;
- richer UI interactions.

But these are **convenience features**, not prerequisites.

For the MVP:

> File system access is the integration layer.

---

# 3. Recommended Project Structure

Do not place large SEEG datasets inside the Obsidian Vault.

Recommended structure:

```text
SEEG_Project/
│
├── AGENTS.md
│
├── data/
│   ├── raw/
│   ├── epochs/
│   └── features/
│
├── analysis/
│   ├── preprocessing.py
│   ├── spectral.py
│   └── decoding.py
│
├── configs/
│   ├── spectral_decoding.yaml
│   └── preprocessing.yaml
│
├── runs/
│   └── spectral_decoding/
│
├── results/
│   ├── arrays/
│   └── tables/
│
└── workspace/                  ← Obsidian Vault
    ├── Research Canvas Design.md
    ├── SEEG_analysis.canvas
    ├── notes/
    └── figures/
```

Suggested setup:

- Obsidian opens `SEEG_Project/workspace/`
- Codex opens `SEEG_Project/`

This allows Codex to access:

- research notes;
- Canvas;
- code;
- configs;
- results;
- data;

while Obsidian does not need to index hundreds of GB of experimental data.

---

# 4. Canvas as a Scientific DAG

The Canvas is conceptually a directed graph.

Nodes represent research objects.

Edges represent relationships or data flow.

Example:

```text
DATA
  ↓
PREPROCESSING
  ↓
FEATURE EXTRACTION
  ↓
DECODING
  ↓
FIGURE
  ↓
INTERPRETATION
  ↓
NEXT QUESTION
```

The Canvas should remain understandable to a human without reading code.

---

# 5. Node Types

Use a limited semantic vocabulary.

Recommended node types:

| Type | Meaning |
|---|---|
| `QUESTION` | Research question |
| `HYPOTHESIS` | Scientific hypothesis |
| `PLAN` | Proposed approach |
| `DATA` | Dataset or folder |
| `ANALYSIS` | Executable scientific analysis |
| `PARAMETERS` | Current analysis configuration |
| `CODE` | Relevant Python/MATLAB/R script |
| `RUN` | Execution instance, usually not shown individually on main Canvas |
| `RESULT` | Numerical/table output |
| `FIGURE` | Figure or visualization |
| `WARNING` | Methodological or data-quality warning |
| `DECISION` | Scientific decision |
| `INTERPRETATION` | Interpretation of results |
| `REFERENCE` | Paper / external evidence |
| `NEXT` | Suggested next research step |

Do not create a separate Canvas node for every small event.

Only information that changes the **scientific state of the project** should become a persistent Canvas object.

---

# 6. Folder Nodes

Obsidian Canvas does not natively treat a directory as one file node.

Do not drag a large folder directly into Canvas.

Instead create a semantic text node:

```text
📁 SEEG Epochs

type: folder
path: ../data/epochs/
role: input
format: .fif
```

For Codex:

```text
type: folder
path: ../data/epochs/
```

means:

> This node represents a real filesystem directory.

Codex should access the directory directly without expanding all files onto the Canvas.

---

# 7. Analysis Nodes

An analysis node describes **what scientific operation should be performed**.

Example:

```text
ANALYSIS

name: Spectral decoding

Extract 5–195 Hz power.
Use 10-Hz frequency bins.
Baseline: -300 to -100 ms.
Perform binary SVM decoding.
Use grouped 5-fold CV.
```

The analysis node should not need to contain implementation details unless scientifically relevant.

The agent may:

1. inspect incoming nodes;
2. read the relevant config;
3. inspect existing analysis code;
4. modify or generate code;
5. execute the analysis;
6. save output artifacts;
7. update the Canvas.

---

# 8. Edge Semantics

Edges can optionally carry instructions.

Example:

```text
DATA
 │
 │ only correct trials
 │ baseline -300~-100 ms
 ▼
ANALYSIS
```

Conceptually:

```text
source node
+
edge instruction
+
target node
```

Recommended interpretation:

> Nodes define what an object **is**.  
> Edges define how objects **relate or flow**.

Large analysis definitions should usually remain in `ANALYSIS` nodes rather than placing all instructions in edge labels.

---

# 9. Pending Figure / Result Nodes

Obsidian does not need a true "empty image" object.

Use a text placeholder:

```text
FIGURE

name: Decoding accuracy
status: pending
id: fig_decoding_01
```

The agent runs the analysis and generates:

```text
workspace/figures/decoding_accuracy.png
```

Then it modifies the Canvas node.

Before:

```json
{
  "id": "fig_decoding_01",
  "type": "text",
  "text": "FIGURE\nDecoding accuracy\nstatus: pending"
}
```

After:

```json
{
  "id": "fig_decoding_01",
  "type": "file",
  "file": "figures/decoding_accuracy.png"
}
```

The agent should preserve:

- node ID;
- x;
- y;
- width;
- height.

This allows a result to visually "appear" in the planned result position.

---

# 10. AI Responses as Canvas Objects

The AI chat window should not be the only place where useful reasoning lives.

Important agent output can be written into the Canvas as structured research objects.

Do **not** attempt to save hidden internal chain-of-thought.

Instead record useful reasoning summaries:

- `PLAN`
- `OBSERVATION`
- `WARNING`
- `DECISION`
- `INTERPRETATION`
- `NEXT`

Example:

```text
QUESTION
   │
   ├── WARNING
   │   Frequency features are strongly correlated.
   │
   ├── SUGGESTION
   │   Compare 10-Hz bins with canonical bands.
   │
   └── NEXT
       Run a sensitivity analysis.
```

Rule:

> Only reasoning that is scientifically useful later should become a persistent Canvas node.

Do not save routine operational chatter such as:

- "I am opening the file";
- "I will inspect the array";
- "Now I will run Python".

---

# 11. Analysis Versions

Do not represent every run as a new Canvas block.

This becomes visually unmanageable.

Instead distinguish:

## Code history

Stored in:

```text
Git
```

Examples:

- refactoring;
- variable renaming;
- bug fixes;
- plotting changes.

## Run history

Stored in:

```text
runs/
```

Every execution can have its own frozen metadata.

## Scientific version history

Stored in:

```text
Canvas
```

Only create a visible scientific version when the change matters scientifically.

Examples:

- random CV → grouped CV;
- canonical bands → 10-Hz bins;
- different baseline definition;
- different statistical framework;
- a method is rejected because of leakage;
- a competing interpretation is explicitly tested.

---

# 12. Main Canvas vs Analysis Canvas

The main Canvas should show the **current project state**.

Example:

```text
RAW DATA
   ↓
PREPROCESSING
   ↓
POWER
   ↓
SPECTRAL DECODING
   ↓
MAIN FIGURE
```

Complex analyses can have their own sub-Canvas.

Example:

```text
spectral_decoding.canvas
```

which contains:

```text
v1
 ↓
v2
 ├─ method A
 └─ method B
       ↓
      v3
       ↓
      v4 ✓
```

This creates two spatial scales:

## Project Canvas

Answers:

> Where is the overall project now?

## Analysis Canvas

Answers:

> How did this analysis develop?

---

# 13. Parameter Management

Parameters should not be stored only inside Python scripts.

Use explicit configuration files.

Example:

```text
configs/spectral_decoding.yaml
```

Example content:

```yaml
frequency:
  min: 5
  max: 195
  step: 10

time:
  window_ms: 100
  step_ms: 20

classifier:
  type: svm
  C: 1.0

cv:
  type: grouped
  folds: 5

baseline:
  start_ms: -300
  end_ms: -100
```

The YAML file is the **source of truth**.

Canvas shows only a readable summary:

```text
PARAMETERS

5–195 Hz / 10-Hz bins
100-ms window / 20-ms step
SVM C=1
Grouped 5-fold CV

config:
../configs/spectral_decoding.yaml
```

---

# 14. Parameter Change Categories

Parameter changes should be classified into three groups.

## 14.1 Computational parameters

Examples:

- `n_jobs`;
- cache size;
- chunk size;
- CPU/GPU selection;
- plotting DPI.

Rule:

> Do not create Canvas versions.

Record only in run metadata or Git.

## 14.2 Tuning parameters

Examples:

- SVM `C`;
- regularization;
- PCA dimension;
- number of estimators.

Rule:

> Treat as model tuning / parameter search.

Do not create a scientific Canvas version for every tested value.

Instead represent the search as one experiment:

```text
PARAMETER SEARCH
   ↓
comparison result
   ↓
selected value
```

## 14.3 Scientific parameters

Examples:

- 0–300 ms vs 100–400 ms;
- baseline definition;
- canonical bands vs fixed 10-Hz bins;
- trial selection;
- CV split strategy;
- statistical test;
- feature definition.

Rule:

> If the parameter changes the scientific interpretation, it may deserve a Canvas branch, comparison, or scientific version.

---

# 15. Current / Candidate / Adopted

Avoid directly overwriting the official analysis whenever experimenting.

Use three states:

## CURRENT

The current accepted configuration.

## CANDIDATE

A configuration currently being tested.

## ADOPTED

A candidate that has been accepted as the new current configuration.

Example:

```text
CURRENT
10-Hz bins
    │
    │ clone
    ▼
CANDIDATE
5-Hz bins
    │
    ▼
RESULT
    │
    ▼
ADOPT / REJECT
```

This makes experimental analysis safer and easier to understand.

---

# 16. Run Tracking

Every actual execution should create a run directory.

Example:

```text
runs/
└── spectral_decoding/
    ├── run_001/
    │   ├── config.yaml
    │   ├── metadata.yaml
    │   ├── result.npz
    │   └── figure.png
    │
    ├── run_002/
    └── run_003/
```

The run should freeze:

- input identity;
- full configuration;
- code commit;
- environment;
- timestamps;
- output files;
- key metrics;
- warnings;
- execution status.

The main Canvas should not display every run.

---

# 17. Reproducibility / Provenance

A researcher should be able to click any final figure and answer:

> Where did this come from?

The full provenance chain should conceptually be:

```text
FIGURE
│
├── input data
├── preprocessing
├── analysis
├── parameters
├── code
├── code commit
├── environment
├── run ID
└── output files
```

Future target:

```text
Reproduce this result
```

should rerun the exact analysis using the frozen run configuration.

---

# 18. Clone & Compare

A major interaction pattern should be:

```text
Clone as experiment
```

Example:

```text
CURRENT
10-Hz bins
   │
   ├── candidate A: 5-Hz bins
   └── candidate B: canonical bands
```

The AI agent runs candidates and generates a structured comparison.

Example:

```text
COMPARE

                 Current   Candidate
frequency step   10 Hz     5 Hz
accuracy         0.63      0.64
peak frequency   105 Hz    110 Hz
peak time        220 ms    215 ms

Conclusion:
Main effect remains stable.
```

Possible actions:

```text
Keep Current
Adopt Candidate
Keep Both
```

The Canvas should emphasize scientific comparison rather than merely generating multiple files.

---

# 19. Scientific Critic Agent

The system should eventually distinguish:

## Executor Agent

Goal:

> Make the requested analysis work.

## Critic Agent

Goal:

> Challenge whether the analysis is scientifically valid.

The Critic should check issues such as:

- train/test leakage;
- non-independent trials;
- stimulus identity leakage;
- class imbalance;
- inappropriate CV;
- multiple comparisons;
- baseline contamination;
- artifact-driven effects;
- circular analysis;
- post-hoc parameter selection;
- small sample instability;
- electrode-level dependence;
- inappropriate statistical assumptions.

Example Canvas output:

```text
WARNING

Possible stimulus-level leakage.

Trials from the same stimulus may enter
both training and test folds.

Suggested check:
Group CV by stimulus identity.
```

The Critic should not silently modify scientific parameters.

---

# 20. Exploratory vs Confirmatory Analysis

The system should explicitly distinguish:

```text
EXPLORATORY
```

from:

```text
CONFIRMATORY
```

If a parameter is selected after inspecting results, record that fact.

Example:

```text
WARNING

100–400 ms window was selected
after comparing multiple windows.

Treat this result as exploratory.
```

This is intended to preserve scientific history, not restrict exploration.

---

# 21. Data Lineage

Derived datasets should have their own provenance.

Example:

```text
RAW SEEG
   ↓
bad-channel detection
   ↓
bipolar rereference
   ↓
epoching
   ↓
artifact rejection
   ↓
epochs_clean_v3
```

A dataset node should eventually be able to report:

- subject count;
- channel count;
- trials before QC;
- trials after QC;
- removed channels;
- removed trials;
- generating script;
- generating config;
- run ID;
- source dataset.

Large binary data should remain outside the Canvas.

Canvas stores identity and lineage, not the raw bytes.

---

# 22. Literature as Part of the Research Graph

Papers should not exist only as disconnected PDFs.

Useful relationships:

```text
Paper A
  │ supports
  ▼
Hypothesis 2

Paper B
  │ contradicts
  ▼
Interpretation 3

Paper C
  │ motivates
  ▼
HFA analysis
```

The long-term goal is a lightweight research knowledge graph.

Canvas edges may be sufficient initially; a separate graph database is not required for the MVP.

---

# 23. Manuscript as a Downstream Product

The paper should eventually become downstream of the analysis graph.

Conceptually:

```text
DATA
 ↓
ANALYSIS
 ↓
RESULT
 ↓
FIGURE
 ↓
INTERPRETATION
 ↓
MANUSCRIPT
```

For a result marked as manuscript-ready:

```text
status: manuscript
figure: Figure 3B
```

the system could maintain:

```text
paper/
├── figures/
│   └── fig3b.svg
├── methods/
│   └── spectral_decoding.md
├── results/
│   └── spectral_decoding.md
└── stats/
    └── fig3b.yaml
```

Methods should be generated from actual:

- configuration;
- run metadata;
- code version;

rather than reconstructed from memory.

If an upstream parameter changes, downstream manuscript items can be marked stale.

---

# 24. Computing Resources Should Become Infrastructure

The researcher should ideally request:

```text
RUN
```

without manually managing:

```text
ssh
tmux
nohup
job IDs
```

Future execution routing may choose:

```text
small QC              → local machine
normal SEEG analysis  → workstation
heavy permutation     → server / HPC
```

The Canvas should only show meaningful execution state.

Example:

```text
RUNNING

spectral decoding
18 / 30 subjects

machine:
workstation
```

A workflow engine such as Snakemake may eventually sit behind the agent, but this is not required for the first MVP.

---

# 25. Canvas Layout Philosophy

Avoid overlapping version blocks.

Overlapping Canvas cards attempt to encode time/version using spatial depth, which Canvas does not handle well.

Use:

```text
x-axis = research progression
y-axis = competing branches / alternatives
```

Recommended main-line pattern:

```text
QUESTION
  ↓
PLAN
  ↓
ANALYSIS
  ↓
RESULT
  ↓
INTERPRETATION
  ↓
NEXT
```

Historical information should have lower visual weight than current state.

Conceptually:

```text
CURRENT ANALYSIS
      │
      └── history
          · v3
          · v2
          · v1
```

Current scientific state is foreground.

Historical states are background.

---

# 26. When Should the Agent Create a New Scientific Version?

Create a visible scientific version only when one of the following is true:

1. the scientific method changed;
2. a competing scientific approach is being tested;
3. an old method was rejected for a scientifically meaningful reason;
4. the interpretation of the result could change;
5. a methodological decision is likely to need explanation in a paper.

Do **not** create a scientific version for:

- typo fixes;
- code cleanup;
- plotting changes;
- variable renaming;
- path corrections;
- `n_jobs` changes;
- refactoring;
- minor implementation details.

---

# 27. Canvas Update Safety

Obsidian and Codex may edit the same `.canvas` file.

Potential issue:

```text
Obsidian edits layout
        ↓
Codex reads old file
        ↓
Obsidian saves
        ↓
Codex writes old structure back
```

MVP rule:

> Avoid moving Canvas nodes while Codex is actively modifying the same Canvas.

Later, introduce a small utility:

```text
tools/canvas.py
```

which performs targeted patch operations instead of rewriting the entire Canvas.

Potential commands:

```text
canvas add-node
canvas connect
canvas update-node
canvas attach-image
canvas add-warning
canvas add-decision
canvas promote-candidate
```

The tool should handle:

- node IDs;
- coordinates;
- layout;
- atomic writes;
- preserving unrelated Canvas content.

---

# 28. Suggested Canvas Utility

Rather than letting the model repeatedly manipulate raw JSON, eventually create:

```text
tools/
└── canvas.py
```

Potential agent-facing API:

```text
add_node(
    type="WARNING",
    title="Class imbalance",
    parent="spectral_decoding"
)
```

```text
attach_image(
    node_id="fig_decoding_01",
    file="figures/decoding_accuracy.png"
)
```

```text
connect(
    source="analysis_01",
    target="fig_decoding_01",
    label="produces"
)
```

This utility should also implement basic automatic layout.

---

# 29. Relationship Between This Document and AGENTS.md

This file is the **design document**.

It explains:

- why the architecture exists;
- the concepts;
- the trade-offs;
- future directions.

`AGENTS.md` should be much shorter.

It should contain only rules Codex must consistently follow.

Example categories for `AGENTS.md`:

```text
Canvas location
Node conventions
Folder node rules
Result node rules
Scientific parameter rules
Version creation rules
Run storage rules
Canvas write safety
Never silently change scientific parameters
```

Do not copy the full design discussion into `AGENTS.md`.

---

# 30. Suggested AGENTS.md Principles

The eventual `AGENTS.md` should include rules such as:

```text
1. Treat workspace/SEEG_analysis.canvas as the current scientific workflow.

2. A text node containing:
   type: folder
   path: ...
   represents a filesystem directory.
   Never expand all files into Canvas nodes.

3. A node with:
   type: result
   status: pending
   is an expected output.

4. After successful figure generation:
   - save figures under workspace/figures/
   - replace/update the pending result node
   - preserve node ID and layout

5. Never silently change:
   - frequency definition
   - baseline
   - epoch window
   - classifier
   - CV strategy
   - statistical test
   - trial inclusion rules

6. Every execution should create reproducible run metadata.

7. Computational parameter changes do not create scientific versions.

8. Tuning parameters belong to parameter-search history.

9. Scientific parameter changes may create comparison branches or new versions.

10. Record important methodological warnings and scientific decisions in Canvas.

11. Do not store hidden chain-of-thought.
    Store concise scientific reasoning summaries only.

12. Preserve current Canvas readability.
    Do not create nodes for routine execution chatter.
```

---

# 31. MVP

Do not build the entire Research OS immediately.

The first prototype only needs:

```text
Obsidian Canvas
+
Codex
+
AGENTS.md
+
canvas.py
+
Python analysis
+
Git
```

## MVP capabilities

### 1. Semantic folder node

Canvas can represent a filesystem folder without expanding its files.

### 2. Analysis node

Agent reads an analysis request from Canvas.

### 3. Parameter config

Agent reads YAML configuration.

### 4. Execute

Agent runs Python/MATLAB analysis.

### 5. Result replacement

Pending result node becomes a figure node after successful execution.

### 6. Reasoning summary

Important warnings / decisions / conclusions are written back into Canvas.

### 7. Run archive

Each execution receives a reproducible run directory.

### 8. Scientific version distinction

Only meaningful methodological changes appear in Canvas history.

---

# 32. Phase 2

After the MVP is comfortable to use, add:

1. Clone & Compare
2. Current / Candidate / Adopted workflow
3. automated run metadata
4. figure provenance
5. targeted Canvas patching
6. Scientific Critic agent
7. exploratory vs confirmatory labeling

---

# 33. Phase 3

Potential later additions:

1. MLflow or equivalent run tracking
2. DataLad or equivalent data lineage
3. Snakemake workflow execution
4. literature relationship graph
5. manuscript synchronization
6. compute-resource routing
7. result invalidation when upstream analysis changes
8. full research timeline / scientific "time machine"

---

# 34. Long-Term Vision

The final system is conceptually:

```text
                 Researcher
                     │
                     ▼
              Obsidian Canvas
          "current research state"
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
    Executor        Critic      Literature
     Agent           Agent        Agent
       │
       ▼
   Workflow Layer
       │
       ▼
 Python / MATLAB / R
       │
       ▼
      RUNS
       │
 ┌─────┼─────────┐
 ▼     ▼         ▼
Git   Run DB   Data lineage
       │
       ▼
Figures / Stats / Tables
       │
       ▼
     Manuscript
```

The researcher should mostly interact with the top layer.

Infrastructure should remain invisible unless needed.

---

# 35. Design Principle

The system should always be able to answer four questions:

> **What is this?**

> **Why was it done this way?**

> **Exactly how was this result produced?**

> **Can the result be reproduced reliably?**

If the workspace can answer these four questions naturally, it has moved beyond an AI coding assistant and toward an **AI-native scientific workspace / Research OS**.

---

# 36. Immediate Next Step

The recommended next implementation step is:

```text
1. Create AGENTS.md
2. Create SEEG_analysis.canvas
3. Define semantic node conventions
4. Implement a minimal tools/canvas.py
5. Test one end-to-end analysis:
   folder node
      ↓
   analysis node
      ↓
   pending figure
      ↓
   Codex executes
      ↓
   figure appears in Canvas
6. Add run metadata
7. Only then expand functionality
```

The success criterion for the MVP is not feature count.

It is:

> Can a researcher visually specify one analysis, let the agent execute it, see the result appear in the planned Canvas location, and later understand exactly how that result was produced?

If yes, the core interaction model is valid.
