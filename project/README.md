# How to run a Geekbot v2 step chat

This folder is the project's memory. A new chat remembers nothing from earlier chats; it knows only what is written down here. So every chat starts by reading these notes, and every step ends by saving them.

**Rule of thumb: one step = one chat.** In that chat you learn one feature, write one piece of it yourself, run the checks, and commit. (A chat may cover a whole phase if you say so at the start; the process is the same.)

This page is the process as you experience it. Claude's side is fixed in [`CLAUDE.md`](../CLAUDE.md), which every chat in this repo loads automatically, and the plan is [`PLAN.md`](PLAN.md) (§2 has the loop). If this page and `CLAUDE.md` ever disagree, `CLAUDE.md` wins; tell Claude and it will fix this page.

**Words you will see**
- **Step IDs:** `S0.1` is Phase 0, step 1. `C1`–`C6` are the CSV analyst track ("C" is for CSV). `P0` and `P1` were planning steps.
- **✍️** is the piece of code you write yourself. **🎮** means the step loads a model on the GPU.
- **⛳** is a review stop: `CP2` (the first results table, after Phase 2) and Gates A, B and C (a checklist review at the end of each big part).

## 1. Before you open a chat
1. Open the chat on the `C:\Work\Geekbot` folder, so `CLAUDE.md` loads.
2. Read **Next up** at the top of [`STATUS.md`](STATUS.md). That is your step. Check that the previous step is committed.
3. If the step has 🎮, start it only when your Gigalogy work isn't using the GPU.
4. Keep a terminal open in the repo with `conda activate geekbot` (the env exists once S0.1 is done).

## 2. The first message
Name the step and let the protocol do the rest:

```text
Start step S0.2. Follow the session protocol in CLAUDE.md.
```

To pick up a step that was left unfinished:

```text
Resume step S1.3. Read STATUS.md and its step log first, then tell me where we stopped.
```

Claude then reads `STATUS.md` → `PLAN.md` → `DECISIONS.md` → the latest step log, tells you where things stand, confirms which step the chat covers, and restates the no-run rule. If the notes and the code disagree, it stops and asks before doing anything.

## 3. What happens in the chat
| # | Stage | What Claude does | What you do |
|---|---|---|---|
| 1 | Lesson | Writes `project/learn/NN-<topic>.md` and teaches the concepts | Read it, explain it back in your own words, ask what is unclear |
| 2 | Design approval | Proposes the files, interfaces and tests, plus the spec of your ✍️ piece. **No code yet** | Approve it or ask for changes |
| 3 | Build | Writes a small diff. Your ✍️ piece is left as a stub with tests that fail until it works. Runs nothing | Skim the diff, ask about anything unclear |
| 4 | ✍️ You write | Waits, or helps (see the hint ladder below) | Write your piece (usually 20–60 lines) |
| 5 | You verify | Gives you the exact commands | Run them in the `geekbot` env and paste the output, errors included. Claude fixes what is its own; you fix your piece |
| 6 | Review | Reviews your code, then asks 3–5 understanding questions | Review Claude's diff, answer the questions in writing |
| 7 | Green light | Gives you the exact git commands | Say "green light", then run them |

Claude never skips stage 2 or stage 7, and never starts the next step on its own.

**Hint ladder** for a stuck ✍️ piece: (1) a hint, (2) an outline, (3) Claude writes it and explains it line by line. If you reach 3, the step log says "written by Claude". A stuck piece never blocks the project.

## 4. Ending a chat
Before you close the chat, check that Claude has:
- [ ] written the step log `project/steps/<id>-<slug>.md` (from [`_TEMPLATE.md`](steps/_TEMPLATE.md))
- [ ] updated `STATUS.md` (status, date, "Next up")
- [ ] added new decisions to `DECISIONS.md`
- [ ] saved the lesson note and listed it in [`learn/README.md`](learn/README.md)
- [ ] given you the git commands

Then you commit. Claude drafts the message in Conventional Commits style, for example `feat(ingest): S1.1 loaders`, ending with a `Co-Authored-By:` line. Code and notes go in **one** commit.
- **Before S0.5:** commit straight on the `v2` branch.
- **From S0.5 on:** each step runs on its own branch `step/<id>-<slug>` and is merged into `v2` through a pull request after CI (the automatic checks on GitHub) passes. S0.5 adds the exact commands here.

Last, tell Claude the commit SHA (`git log -1 --format=%h`). Claude records it in `STATUS.md`, at the latest at the start of the next chat.

At the end of each phase, the last step's log also gets a short **Phase look-back**: planned vs actual, and what to change.

## 5. If a chat stops early
The notes are the memory, so save them before you leave:

```text
Save our progress to the step log and STATUS.md, then tell me how to resume.
```

The status in `STATUS.md` tells the next chat where to pick up:

| Status | The next chat resumes at |
|---|---|
| `design` | the lesson or the design approval (stages 1–2) |
| `build` | the build and your ✍️ piece (stages 3–4) |
| `verify` | your command output and the review (stages 5–6) |
| `green-lit` | nothing: the step is done; Claude records the commit SHA |
| `blocked` | the blocker written in the step log |

## 6. Other kinds of chats
- **Changing the plan.** Say what you want to change. Claude proposes it in plain words. Once you approve, it updates `PLAN.md`, adds a line to the plan's revision history (§4) and an entry to `DECISIONS.md`. Never silently.
- **Review stops** (CP2, Gates A, B, C). You run the checklist in `PLAN.md` §11, Claude reviews it with you, and the phase look-back is written.
- **External reviews** (ChatGPT, Codex). Paste the review as it is. Claude saves it in `project/reviews/` with a table showing what was adopted and why.
- **Question chats.** No step; nothing to commit unless a note changes.

## 7. Rules that don't change
This is the short version; the full text is in `CLAUDE.md`.
- **Claude runs nothing on this PC:** no commands, installs, tests, model calls or git. You run everything and paste the output. The PC also runs your CSV project.
- **Commands run in the `geekbot` conda env, one command per line.** (`&&` doesn't work in Windows PowerShell 5.1.)
- **One step at a time.** No code before you approve the design, no commit before your green light, nothing from the Parked list in `PLAN.md` §10.
- **No model over 12B, speed is a hard requirement, and a number is reported only if it was measured.** Which models are used is set in `CLAUDE.md` (rule 4) and `PLAN.md` §6.5.
- **`PLAN.md` changes only with your approval.**

## 8. Learn actively (suggested, not required)
The lesson notes end with interview questions and model answers. Reading them is not the same as knowing them. A few minutes per stage makes the difference; drop any habit that slows you down.

| Stage | Do this |
|---|---|
| Lesson | Before reading, write 2 lines on what you think the idea is. After, close the note and explain it back. Where you stall is what to re-read |
| Design approval | For each choice ask: "What's the second-best option, and why not that?" |
| ✍️ You write | Write it before asking for a hint. Once it passes, break one line on purpose and predict which test fails |
| You verify | Predict the output before you run. A surprise is the best moment to learn, so write down why it happened |
| Review | Answer the questions in writing before you read the model answers in the lesson note |

Write your answers in the step log's *Review + walkthrough Q&A* section. Over time they become your own answer bank for interviews.

## Where things live
| Path | What it holds |
|---|---|
| [`../CLAUDE.md`](../CLAUDE.md) | the rules and session protocol Claude follows |
| [`PLAN.md`](PLAN.md) | the plan, the single source of truth |
| [`STATUS.md`](STATUS.md) | the progress tracker and "Next up" |
| [`DECISIONS.md`](DECISIONS.md) | the decision log |
| [`steps/`](steps/README.md) | one log per step |
| [`learn/`](learn/README.md) | lesson notes and their index |
| [`reviews/`](reviews/README.md) | external reviews and how each point was resolved |
| [`sources/`](sources/) | outside documents the plan builds on |

The `README.md` at the repo root is still v1's. It is rewritten at step S5.5.
