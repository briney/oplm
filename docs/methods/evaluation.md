# Evaluation

What each eval measures and how to run it: [../evaluation.md](../evaluation.md).

| Choice | Where the value lives | Evidence |
|---|---|---|
| Structure eval sets and their construction | [../evaluation.md](../evaluation.md), *Published structure eval sets* | Provenance, curation rules and counts: [../data.md](../data.md), *Structure evaluation sets*. No run compares alternative sets |
| Coevolution eval | [../evaluation.md](../evaluation.md) | Reference build and its checks: [../data.md](../data.md), *Coevolution data*. First results: [coevolution-eval-readout](evidence/coevolution-eval-readout/README.md) |
| Antibody CDR3 eval | [../evaluation.md](../evaluation.md), *antibody_cdr3* | Per-position masking was chosen without a comparison run (whole-CDR3 masking and 15% random masking were rejected by argument; PR #95, *Why this masking*). First use: [antibody-fraction](evidence/antibody-fraction/README.md) |
| Decisions are judged after a decay, not on the stable phase | [../mup-lr.md](../mup-lr.md), *Why and what was ruled out* | [base-lr-funnel](evidence/base-lr-funnel/README.md): the learning-rate ranking reverses after the decay, over three seeds |
