# Methods

The source material for the methods section of the paper: what the production models are, why
each choice was made, and the results behind each choice. It is a re-implementation plan, not a
history. Git history is the history.

## Layout

- **Topic pages** list every choice a reader might question, where its value lives, and the
  evidence for it. They do not restate how things work; they link to the doc that does.
  - [architecture.md](architecture.md)
  - [data.md](data.md)
  - [optimization.md](optimization.md): optimizer, learning rate, schedule, batch size.
  - [evaluation.md](evaluation.md)
  - [fold.md](fold.md)
- **Evidence records** live in `evidence/<slug>/`, one directory per decision: a `README.md`, a
  `results.csv`, a `configs/` directory with each arm's resolved config, and a `curves.csv` only
  when the figure is a curve.

## Rules

1. **Describe production today.** Topic pages are rewritten in place. When a choice changes, its
   row changes and the old record stays only if it still explains the choice or documents a
   rejected alternative; otherwise it is deleted.
2. **Write the record when the result is read off the run**, not later. Ties and rejected
   alternatives get records too.
3. **A PR that changes a production value updates the topic page and the record in the same PR.**
4. **Point to configs instead of copying numbers.** Topic pages name the YAML file and field.
   Numbers appear in evidence records, because they are results.
5. **Unmatched evidence goes in anyway.** Say what differs between arms under Caveats, set the
   status to `needs matched rerun`, and put the rerun on the board.
6. **One fact, one home.** How a thing works stays in its home doc under `docs/`. Why it was
   chosen, with the numbers, lives here.

## Evidence record

Copy this into `evidence/<slug>/README.md`. Keep all six headings, in this order.

```markdown
# <Decision, as a statement: "Antibody sequences are 1.5% of the decay-phase mix">

**Status:** settled | provisional | needs matched rerun

## Question

What was being decided, and what would have changed had it gone the other way.

## Arms

What was compared and what was held fixed (model size, data, tokens per step, steps, seed).

## Result

The verdict in one sentence, then the table (the same rows as `results.csv`, the metrics that
decided it).

## Runs

Where the raw runs are, beyond the IDs in `results.csv`: W&B project, cluster output paths,
the PR or issue with the discussion.

## Caveats

Anything unmatched between arms, anything measured at a proxy scale, anything not yet rerun.
```

## CSV conventions

A record that reuses another record's runs points its `config` column at that record's `configs/`
instead of copying the files.

`results.csv` has one row per run, with these columns first and the metrics after them:

| Column | Content |
|---|---|
| `arm` | Short name of the arm, as used in the record's table. |
| `wandb_run` | W&B run ID (the 8-character ID, not the display name). |
| `job_id` | Slurm job ID, empty if none. |
| `commit` | Commit the run was launched from. |
| `config` | `configs/<arm>.yaml`: the run's resolved config (`output/config.yaml` from the run directory), committed next to the CSV. |
| `step` | Step the metrics were read at. |

Metric columns use the W&B metric name where the number was read from W&B, and a descriptive
name otherwise (the record says which). A record may add another CSV for results that are not training runs (an eval-only sweep, say);
its README names the file and its columns.

`curves.csv` is long format: `arm`, `wandb_run`, `step`,
`metric`, `value`, downsampled to what a figure needs.

## Status of a row

The Evidence column of a topic page holds a link to the record once it exists. Until then it reads
`to backfill`, with where the numbers are today.
