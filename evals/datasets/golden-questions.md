# Golden questions

The hand-written test questions for the golden corpus (`evals/corpus/`). They check how Geekbot behaves on **our** documents: "system-specific behavioral evaluation" (PLAN.md §9). They say nothing about retrieval quality in general; the BEIR benchmarks cover that.

**This is golden v0: written by Claude, provisional (PLAN rev 18, D-067).** The corpus is 15 original documents about a fictional studio, Brightwater Studio, and its AI assistant "Lantern". The author of the documents, of these questions and of the system is one, so v0 is easier than real use: its results are reported as "Claude-authored golden v0, provisional" and never as a headline. Your own questions go in a separate `evals/datasets/golden-user.jsonl` and are scored on their own. Every item below carries the flag "v0 (Claude-authored, provisional)" in its notes, and a test checks that.

**This file is the source of truth.** `geekbot eval import` reads it, checks it, and writes `evals/datasets/golden.jsonl` (step S2.4). Edit this file, never the JSONL: the JSONL is overwritten on every import. Blocks left empty are skipped.

**Target: ~100 questions**

| Type | Target | What it tests |
|---|---|---|
| Single-hop | ~50 | the answer is in one place in one document |
| Multi-hop | ~15 | the answer needs facts from two or more places |
| Multi-turn | ~15 | a follow-up that only makes sense after an earlier question |
| Unanswerable | ~20 | the documents don't contain the answer, so Geekbot should say it doesn't know |

**What each field is for**
- **Question:** written the way a real user would ask it, not copied from the document.
- **Where the answer is:** the file and, for you, the page or section, for every place needed. This checks whether search finds the right passages, and whether the answer cites them.
- **Must include:** the key facts a correct answer has to contain, as short phrases that occur in the documents.
- **Notes:** anything tricky (optional).

**How to write the two list fields** (the importer reads them literally)
- *Where the answer is*: one place per line (a `- ` bullet is fine) or separated by `;`. Write each place as `path/in/corpus.md (page or section)`. The path is **relative to the `evals/corpus/` folder**, with forward slashes, and must be a real file there; the part in parentheses is for you and is optional.
- *Must include*: short phrases, one per line or separated by `;`.
- A field can continue on the next lines. Don't add fields that are not in the blocks below.

**The ID decides the type** (S single-hop, M multi-hop, T multi-turn, U unanswerable), not the section heading. Number them in order and don't reuse an ID.

**The split.** About 40% of each type becomes *development* items (used to pick thresholds and depths) and the rest *test* items (used for the reported score), decided by a fixed rule from the ID (D-065). Finish the set before anything is tuned, and don't renumber after that.

**Qrels limitation, stated.** The documents overlap on purpose, so a fact is sometimes stated in more than one file. A single-hop item lists one source (its main one); a multi-hop item lists every file that states a fact the answer needs. Document-level recall can therefore under-count a single-hop answer found in a second file.

A worked example (this fenced block is ignored by the importer):

```
### S-017
- **Question:** How long does the router take to finish a factory reset?
- **Where the answer is:** router/manual.md (Reset > Factory reset)
- **Must include:** ten seconds; amber light
- **Notes:** the manual says "about ten seconds"; the question avoids the word "factory".
```

---

## Single-hop (~50)

### S-001
- **Question:** How much video memory does the graphics card in the Lantern computer have?
- **Where the answer is:** ops/hardware-setup.md (The machine)
- **Must include:** 12 GB of video memory
- **Notes:** v0 (Claude-authored, provisional)

### S-002
- **Question:** How fast does the assistant produce text, counted in tokens?
- **Where the answer is:** ops/hardware-setup.md (Speed)
- **Must include:** 38 tokens per second
- **Notes:** v0 (Claude-authored, provisional)

### S-003
- **Question:** In the first speed test, how long did the slowest 5 percent of requests wait before the first word of an answer?
- **Where the answer is:** ops/hardware-setup.md (Speed)
- **Must include:** 3.4 seconds
- **Notes:** v0 (Claude-authored, provisional); asks for the 95th percentile without using the term

### S-004
- **Question:** What context length did the IT contractor set to bring response times down?
- **Where the answer is:** ops/hardware-setup.md (Speed)
- **Must include:** 4,096 tokens
- **Notes:** v0 (Claude-authored, provisional)

### S-005
- **Question:** For how long is the battery unit rated to keep the computer running at normal load?
- **Where the answer is:** ops/hardware-setup.md (Power)
- **Must include:** about 20 minutes
- **Notes:** v0 (Claude-authored, provisional); the pre-launch test and the real power cut gave other figures

### S-006
- **Question:** How soon must the most serious kind of incident be acknowledged?
- **Where the answer is:** ops/incident-runbook.md (How serious is it)
- **Must include:** within 15 minutes
- **Notes:** v0 (Claude-authored, provisional)

### S-007
- **Question:** What is the deadline for telling a client about a medium-severity incident that involved their data?
- **Where the answer is:** ops/incident-runbook.md (How serious is it)
- **Must include:** within 72 hours
- **Notes:** v0 (Claude-authored, provisional)

### S-008
- **Question:** Who talks to the client when there is an incident?
- **Where the answer is:** ops/incident-runbook.md (What to do, in order)
- **Must include:** Maya speaks to the client
- **Notes:** v0 (Claude-authored, provisional)

### S-009
- **Question:** How long after an incident has been contained does the review have to take place?
- **Where the answer is:** ops/incident-runbook.md (What to do, in order)
- **Must include:** within 5 working days
- **Notes:** v0 (Claude-authored, provisional)

### S-010
- **Question:** Where is a minor incident, such as an outage, talked through?
- **Where the answer is:** ops/incident-runbook.md (How serious is it)
- **Must include:** discussed at the weekly project meeting
- **Notes:** v0 (Claude-authored, provisional)

### S-011
- **Question:** When the deletion job was tested before launch, which file ages were removed and which were left?
- **Where the answer is:** ops/security-checklist.md (the retention job row)
- **Must include:** aged 91 days were deleted; aged 89 days were kept
- **Notes:** v0 (Claude-authored, provisional)

### S-012
- **Question:** How long did the computer keep running on battery in the pre-launch test?
- **Where the answer is:** ops/security-checklist.md (the power backup row)
- **Must include:** 21 minutes
- **Notes:** v0 (Claude-authored, provisional); the real power cut in July gave a different figure

### S-013
- **Question:** How many of the staff had been through training by the time the checklist was signed off?
- **Where the answer is:** ops/security-checklist.md (the training row)
- **Must include:** 12 of 12
- **Notes:** v0 (Claude-authored, provisional)

### S-014
- **Question:** Why was a second person asked to attend the first incident review?
- **Where the answer is:** ops/security-checklist.md (Open points at sign-off)
- **Must include:** second person should learn the runbook
- **Notes:** v0 (Claude-authored, provisional)

### S-015
- **Question:** Which file does the nightly deletion program write its messages to?
- **Where the answer is:** ops/log_cleanup.py (the constants at the top)
- **Must include:** /var/log/lantern/cleanup.log
- **Notes:** v0 (Claude-authored, provisional)

### S-016
- **Question:** Does the deletion program remove a log that is exactly as old as the limit?
- **Where the answer is:** ops/log_cleanup.py (the expired function)
- **Must include:** old is kept
- **Notes:** v0 (Claude-authored, provisional)

### S-017
- **Question:** Which fields are recorded for every incident in the incident table?
- **Where the answer is:** ops/incident-log.csv (the columns)
- **Must include:** incident_id; severity; resolved_date; hours_to_resolve
- **Notes:** v0 (Claude-authored, provisional); a table is indexed as a schema card only (D-051)

### S-018
- **Question:** What kind of number is the time-to-fix column in the incident table?
- **Where the answer is:** ops/incident-log.csv (the columns)
- **Must include:** hours_to_resolve; decimal
- **Notes:** v0 (Claude-authored, provisional); schema card only

### S-019
- **Question:** Which sheets does the workbook about tools and costs have?
- **Where the answer is:** project/tools-and-costs.xlsx (the sheets)
- **Must include:** sheet Tools; sheet Budget
- **Notes:** v0 (Claude-authored, provisional); schema card only

### S-020
- **Question:** In the tools sheet, which column shows who looks after each tool?
- **Where the answer is:** project/tools-and-costs.xlsx (Tools sheet)
- **Must include:** owner
- **Notes:** v0 (Claude-authored, provisional); schema card only

### S-021
- **Question:** Who is responsible for the AI use policy?
- **Where the answer is:** policy/ai-use-policy.docx (the opening lines)
- **Must include:** Owner: Daniel Reyes
- **Notes:** v0 (Claude-authored, provisional)

### S-022
- **Question:** Do the AI rules cover outside contractors as well as employees?
- **Where the answer is:** policy/ai-use-policy.docx (Purpose and scope)
- **Must include:** all 12 members of staff and to contractors
- **Notes:** v0 (Claude-authored, provisional)

### S-023
- **Question:** How frequently is the AI use policy looked at again?
- **Where the answer is:** policy/ai-use-policy.docx (Approval and review)
- **Must include:** every six months
- **Notes:** v0 (Claude-authored, provisional)

### S-024
- **Question:** What address do I type to open Lantern, and does it work from home?
- **Where the answer is:** policy/onboarding-letter.docx (the opening paragraphs)
- **Must include:** lantern.office.lan; it does not work from home
- **Notes:** v0 (Claude-authored, provisional)

### S-025
- **Question:** What follow-up does the operations lead promise a new user?
- **Where the answer is:** policy/onboarding-letter.docx (the closing line)
- **Must include:** After your first four weeks I will ask how it is going
- **Notes:** v0 (Claude-authored, provisional)

### S-026
- **Question:** What small first task does the welcome letter suggest trying?
- **Where the answer is:** policy/onboarding-letter.docx (Your first week)
- **Must include:** summary of last week
- **Notes:** v0 (Claude-authored, provisional)

### S-027
- **Question:** Which three folders hold the data that Lantern stores?
- **Where the answer is:** policy/data-handling-standard.txt (section 1.2)
- **Must include:** /srv/lantern/uploads; /srv/lantern/chat_logs; /srv/lantern/audit
- **Notes:** v0 (Claude-authored, provisional)

### S-028
- **Question:** If I delete a file from Lantern, can I get it back from a recycle bin?
- **Where the answer is:** policy/data-handling-standard.txt (section 6.1)
- **Must include:** keeps no recycle bin
- **Notes:** v0 (Claude-authored, provisional)

### S-029
- **Question:** Are the backups changed when a whole workspace is deleted?
- **Where the answer is:** policy/data-handling-standard.txt (section 6.2)
- **Must include:** Backups are not edited
- **Notes:** v0 (Claude-authored, provisional)

### S-030
- **Question:** Where is the key for the weekly cloud backup kept?
- **Where the answer is:** policy/data-handling-standard.txt (section 5.2)
- **Must include:** office safe
- **Notes:** v0 (Claude-authored, provisional)

### S-031
- **Question:** What is the longest a conversation can exist anywhere, backups included?
- **Where the answer is:** policy/data-handling-standard.txt (section 3.3)
- **Must include:** 120 days
- **Notes:** v0 (Claude-authored, provisional); the figure is the sum of two retention periods, stated in one place

### S-032
- **Question:** Who gave the presentation at the project kickoff?
- **Where the answer is:** project/rollout-kickoff.pptx (slide 1)
- **Must include:** Presenter: Daniel Reyes
- **Notes:** v0 (Claude-authored, provisional)

### S-033
- **Question:** According to the kickoff slides, what were employees already doing with AI before Lantern?
- **Where the answer is:** project/rollout-kickoff.pptx (slide 2)
- **Must include:** Staff already try outside AI tools on their own
- **Notes:** v0 (Claude-authored, provisional)

### S-034
- **Question:** Is Lantern supposed to take decisions that affect staff?
- **Where the answer is:** project/rollout-kickoff.pptx (slide 3)
- **Must include:** will not decide anything about people
- **Notes:** v0 (Claude-authored, provisional)

### S-035
- **Question:** Why did a designer argue against keeping chats for only 30 days?
- **Where the answer is:** project/meeting-notes.txt (10 March 2025)
- **Must include:** three months later
- **Notes:** v0 (Claude-authored, provisional)

### S-036
- **Question:** What was changed after a client draft ended up in the wrong workspace, and on what date?
- **Where the answer is:** project/meeting-notes.txt (5 May 2025)
- **Must include:** banner; 24 April
- **Notes:** v0 (Claude-authored, provisional)

### S-037
- **Question:** Why did the nightly deletion job skip a night in May?
- **Where the answer is:** project/meeting-notes.txt (12 June 2025)
- **Must include:** cron entry was lost in a system update
- **Notes:** v0 (Claude-authored, provisional)

### S-038
- **Question:** What did the security officer conclude about client data in the case of the outside AI tool?
- **Where the answer is:** project/meeting-notes.txt (12 June 2025)
- **Must include:** no client data left the studio in a form that could be read by others
- **Notes:** v0 (Claude-authored, provisional)

### S-039
- **Question:** How long was the refresher session planned to be after the outside AI tool incident?
- **Where the answer is:** project/meeting-notes.txt (12 June 2025)
- **Must include:** 20 minute refresher
- **Notes:** v0 (Claude-authored, provisional)

### S-040
- **Question:** What would one client's agreement have required before the hosted option could be used?
- **Where the answer is:** project/vendor-memo.pdf (Option B)
- **Must include:** Corvid Foods would have to agree to a change in its NDA
- **Notes:** v0 (Claude-authored, provisional)

### S-041
- **Question:** By how much would the local option have cost more than the cloud subscription in the first year?
- **Where the answer is:** project/vendor-memo.pdf (What matters most)
- **Must include:** 14,090
- **Notes:** v0 (Claude-authored, provisional)

### S-042
- **Question:** What would running the local option cost in its second year?
- **Where the answer is:** project/vendor-memo.pdf (Option C)
- **Must include:** 14,928
- **Notes:** v0 (Claude-authored, provisional)

### S-043
- **Question:** What is the largest file I can upload to Lantern?
- **Where the answer is:** staff/staff-faq.html (Which files can I upload)
- **Must include:** up to 50 MB
- **Notes:** v0 (Claude-authored, provisional)

### S-044
- **Question:** Can Lantern read the spreadsheets and pictures that I upload?
- **Where the answer is:** staff/staff-faq.html (Which files can I upload)
- **Must include:** Spreadsheets and pictures are not read
- **Notes:** v0 (Claude-authored, provisional)

### S-045
- **Question:** Roughly how many characters make up one token?
- **Where the answer is:** staff/glossary.md (Token)
- **Must include:** about four characters
- **Notes:** v0 (Claude-authored, provisional)

### S-046
- **Question:** On which weekday and in which hour was Lantern busiest during the pilot?
- **Where the answer is:** analysis/usage-analysis.ipynb (Headline numbers)
- **Must include:** Tuesday; 10:00 to 11:00
- **Notes:** v0 (Claude-authored, provisional)

### S-047
- **Question:** What share of the pilot prompts was for writing emails to clients?
- **Where the answer is:** analysis/usage-analysis.ipynb (What people used it for)
- **Must include:** 27 percent
- **Notes:** v0 (Claude-authored, provisional)

### S-048
- **Question:** Why might the pilot's usage figures be higher than the studio average?
- **Where the answer is:** analysis/usage-analysis.ipynb (Caveats)
- **Must include:** heavy computer users
- **Notes:** v0 (Claude-authored, provisional)

### S-049
- **Question:** How much RAM and what kind of storage does the Lantern computer have?
- **Where the answer is:** ops/hardware-setup.md (The machine)
- **Must include:** 64 GB of RAM; 2 TB NVMe SSD
- **Notes:** v0 (Claude-authored, provisional)

### S-050
- **Question:** Which version of the data handling standard is the latest, and when was it issued?
- **Where the answer is:** policy/data-handling-standard.txt (the title block)
- **Must include:** Version 1.1; 14 April 2025
- **Notes:** v0 (Claude-authored, provisional)

## Multi-hop (~15)

### M-001
- **Question:** What phone number would I use to reach the person who approved the AI use policy?
- **Where the answer is:**
  - policy/ai-use-policy.docx (Approval and review)
  - ops/security-checklist.md (the policy row)
  - ops/incident-runbook.md (Who to call)
- **Must include:** Maya Okafor approved this policy; signed by Maya Okafor; 07700 900103
- **Notes:** v0 (Claude-authored, provisional); who approved is in two files, her number in a third

### M-002
- **Question:** How did the battery run-time in the real power cut compare with the pre-launch test?
- **Where the answer is:**
  - ops/hardware-setup.md (Power)
  - ops/security-checklist.md (the power backup row)
- **Must include:** held for 22 minutes; ran for 21 minutes
- **Notes:** v0 (Claude-authored, provisional); two look-alike figures in two files

### M-003
- **Question:** Was the Lantern computer ordered before or after the AI use policy was approved, and by how many days?
- **Where the answer is:**
  - ops/hardware-setup.md (The machine)
  - policy/ai-use-policy.docx (Approval and review)
  - project/rollout-kickoff.pptx (Timeline)
- **Must include:** ordered on 17 March 2025; approved this policy on 21 March 2025; 17 March: workstation ordered
- **Notes:** v0 (Claude-authored, provisional); answer: before, by 4 days

### M-004
- **Question:** Could all four pilot users have chatted with Lantern at exactly the same moment without anyone having to wait?
- **Where the answer is:**
  - ops/hardware-setup.md (How many people at once)
  - staff/glossary.md (Pilot group)
- **Must include:** Up to 3 people; four people who tested Lantern
- **Notes:** v0 (Claude-authored, provisional); answer: no, the fourth would queue

### M-005
- **Question:** How quickly should the incident where a client draft landed in the wrong workspace have been acknowledged?
- **Where the answer is:**
  - project/meeting-notes.txt (5 May 2025)
  - ops/incident-runbook.md (How serious is it)
- **Must include:** level S2: INC-003; within 4 hours
- **Notes:** v0 (Claude-authored, provisional); the level is in one file, the response time in another

### M-006
- **Question:** If client data had been involved in the June incident with the outside AI tool, within what time would the client have had to be told?
- **Where the answer is:**
  - project/meeting-notes.txt (12 June 2025)
  - ops/incident-runbook.md (How serious is it)
- **Must include:** Level S2; within 72 hours
- **Notes:** v0 (Claude-authored, provisional)

### M-007
- **Question:** What happened to chat logs that passed 90 days on the night the cleanup job did not run in May?
- **Where the answer is:**
  - project/meeting-notes.txt (12 June 2025)
  - policy/data-handling-standard.txt (section 3.2)
  - ops/log_cleanup.py (the opening docstring)
- **Must include:** not running for one night; removed by the nightly retention job; deletes chat logs older than 90 days
- **Notes:** v0 (Claude-authored, provisional); answer: they stayed one more night and were removed at the next run

### M-008
- **Question:** Would the full backup restore that was tested in May fit on the Lantern computer's disk, and how long did it take?
- **Where the answer is:**
  - ops/hardware-setup.md (The machine)
  - ops/security-checklist.md (the backup restore row)
  - policy/data-handling-standard.txt (section 5.3)
- **Must include:** 2 TB NVMe SSD; 310 GB took 47 minutes; restored 310 GB in 47 minutes
- **Notes:** v0 (Claude-authored, provisional)

### M-009
- **Question:** How much does the weekly offsite backup cost each month, and in which region is it stored?
- **Where the answer is:**
  - project/rollout-kickoff.pptx (Budget)
  - project/vendor-memo.pdf (Option C)
  - policy/data-handling-standard.txt (section 5.2)
- **Must include:** backup storage £9; 9 of cloud storage for backups; EU region
- **Notes:** v0 (Claude-authored, provisional)

### M-010
- **Question:** Roughly how many characters of text fit into the context length that Priya set?
- **Where the answer is:**
  - ops/hardware-setup.md (Speed)
  - staff/glossary.md (Token)
- **Must include:** 4,096 tokens; about four characters
- **Notes:** v0 (Claude-authored, provisional); answer: about 16,000 characters

### M-011
- **Question:** When was the action to shorten the context length, agreed at the pilot start, carried out, and to what value?
- **Where the answer is:**
  - project/meeting-notes.txt (7 April 2025)
  - ops/hardware-setup.md (Speed)
- **Must include:** shorten the context length; shortened the context length to 4,096 tokens
- **Notes:** v0 (Claude-authored, provisional)

### M-012
- **Question:** How much longer was the original training session than the planned refresher?
- **Where the answer is:**
  - policy/onboarding-letter.docx (the opening paragraphs)
  - policy/ai-use-policy.docx (Rules for everyone)
  - project/meeting-notes.txt (12 June 2025)
- **Must include:** It takes 45 minutes; 45 minute training session; 20 minute refresher
- **Notes:** v0 (Claude-authored, provisional); answer: 25 minutes

### M-013
- **Question:** Which two people hold administrator rights, and what is the first thing one of them does when an incident starts?
- **Where the answer is:**
  - policy/data-handling-standard.txt (section 2.3)
  - ops/incident-runbook.md (What to do, in order)
- **Must include:** two administrators, Priya Nair and Tomasz Lewandowski; disables the affected account
- **Notes:** v0 (Claude-authored, provisional)

### M-014
- **Question:** Who presented at the kickoff, and was Leo Park in the meeting?
- **Where the answer is:**
  - project/rollout-kickoff.pptx (slide 1)
  - project/meeting-notes.txt (3 March 2025)
- **Must include:** Presenter: Daniel Reyes; Present: Maya Okafor, Daniel Reyes, Priya Nair, Tomasz Lewandowski, Hannah Frost, Samir Haddad
- **Notes:** v0 (Claude-authored, provisional); answer: Daniel Reyes; no, Leo Park is not on the list

### M-015
- **Question:** Was disk encryption switched on during the weeks in which the computer was being set up?
- **Where the answer is:**
  - ops/hardware-setup.md (The machine)
  - ops/security-checklist.md (the full-disk encryption row)
  - project/rollout-kickoff.pptx (Timeline)
- **Must include:** between 28 and 31 March; 29 March; 28 to 31 March
- **Notes:** v0 (Claude-authored, provisional); answer: yes, on 29 March

## Multi-turn (~15)

### T-001
- **Turn 1 (the earlier question):** How long does Lantern keep chat history?
- **Turn 2 (the follow-up that gets graded):** And the logs that only record who used it?
- **Where the answer is:** policy/ai-use-policy.docx (Retention)
- **Must include:** 365 days
- **Notes:** v0 (Claude-authored, provisional)

### T-002
- **Turn 1 (the earlier question):** What do the incident levels mean, starting with the worst?
- **Turn 2 (the follow-up that gets graded):** For that one, how fast must the client be told?
- **Where the answer is:** ops/incident-runbook.md (How serious is it)
- **Must include:** within 24 hours
- **Notes:** v0 (Claude-authored, provisional)

### T-003
- **Turn 1 (the earlier question):** Who do I call when Lantern is broken?
- **Turn 2 (the follow-up that gets graded):** Which days are they on call?
- **Where the answer is:**
  - ops/incident-runbook.md (Who to call)
  - staff/staff-faq.html (Lantern is slow)
  - policy/onboarding-letter.docx (Who to ask)
- **Must include:** Mondays and Thursdays
- **Notes:** v0 (Claude-authored, provisional)

### T-004
- **Turn 1 (the earlier question):** Why did the studio choose a computer in its own office instead of a cloud subscription?
- **Turn 2 (the follow-up that gets graded):** Which clients does that concern?
- **Where the answer is:**
  - project/vendor-memo.pdf (What matters most)
  - project/rollout-kickoff.pptx (Why we are doing this)
  - staff/glossary.md (NDA)
- **Must include:** Harbour and Pine and Corvid Foods
- **Notes:** v0 (Claude-authored, provisional)

### T-005
- **Turn 1 (the earlier question):** How much did the Lantern computer cost?
- **Turn 2 (the follow-up that gets graded):** And the battery unit that goes with it?
- **Where the answer is:**
  - ops/hardware-setup.md (The machine)
  - project/vendor-memo.pdf (Option C)
  - project/rollout-kickoff.pptx (Budget)
- **Must include:** £180
- **Notes:** v0 (Claude-authored, provisional)

### T-006
- **Turn 1 (the earlier question):** When did the pilot run?
- **Turn 2 (the follow-up that gets graded):** How many people took part?
- **Where the answer is:**
  - project/rollout-kickoff.pptx (Timeline)
  - staff/glossary.md (Pilot group)
- **Must include:** four people
- **Notes:** v0 (Claude-authored, provisional)

### T-007
- **Turn 1 (the earlier question):** What kinds of file can I upload to Lantern?
- **Turn 2 (the follow-up that gets graded):** How big can they be?
- **Where the answer is:** staff/staff-faq.html (Which files can I upload)
- **Must include:** 50 MB
- **Notes:** v0 (Claude-authored, provisional)

### T-008
- **Turn 1 (the earlier question):** What does the nightly cleanup job do?
- **Turn 2 (the follow-up that gets graded):** And what does it do while a legal hold is in place?
- **Where the answer is:**
  - ops/log_cleanup.py (the opening docstring)
  - policy/data-handling-standard.txt (section 4.1)
- **Must include:** deletes nothing
- **Notes:** v0 (Claude-authored, provisional)

### T-009
- **Turn 1 (the earlier question):** Who is in charge when an incident happens?
- **Turn 2 (the follow-up that gets graded):** What number do I phone him on?
- **Where the answer is:** ops/incident-runbook.md (Who to call)
- **Must include:** 07700 900101
- **Notes:** v0 (Claude-authored, provisional)

### T-010
- **Turn 1 (the earlier question):** In the first speed test, what was the 95th percentile wait for the first word?
- **Turn 2 (the follow-up that gets graded):** And after the fix?
- **Where the answer is:** ops/hardware-setup.md (Speed)
- **Must include:** 2.8 seconds
- **Notes:** v0 (Claude-authored, provisional)

### T-011
- **Turn 1 (the earlier question):** Can I use an outside AI website for client work?
- **Turn 2 (the follow-up that gets graded):** What if the text is already public?
- **Where the answer is:** staff/staff-faq.html (May I use an outside AI website)
- **Must include:** only for green material
- **Notes:** v0 (Claude-authored, provisional)

### T-012
- **Turn 1 (the earlier question):** How many incidents happened during the pilot?
- **Turn 2 (the follow-up that gets graded):** Which of them was serious?
- **Where the answer is:** project/meeting-notes.txt (5 May 2025)
- **Must include:** INC-003
- **Notes:** v0 (Claude-authored, provisional)

### T-013
- **Turn 1 (the earlier question):** When was the AI use policy approved?
- **Turn 2 (the follow-up that gets graded):** And when does it come up for review?
- **Where the answer is:** policy/ai-use-policy.docx (Approval and review)
- **Must include:** 21 September 2025
- **Notes:** v0 (Claude-authored, provisional)

### T-014
- **Turn 1 (the earlier question):** What does it cost each month to keep Lantern running?
- **Turn 2 (the follow-up that gets graded):** How much of that is the support retainer?
- **Where the answer is:**
  - project/vendor-memo.pdf (Option C)
  - project/rollout-kickoff.pptx (Budget)
- **Must include:** 1,200
- **Notes:** v0 (Claude-authored, provisional)

### T-015
- **Turn 1 (the earlier question):** Where does the nightly backup go?
- **Turn 2 (the follow-up that gets graded):** And the weekly one?
- **Where the answer is:**
  - ops/hardware-setup.md (Scheduled jobs)
  - policy/data-handling-standard.txt (section 5)
- **Must include:** cloud storage in an EU region
- **Notes:** v0 (Claude-authored, provisional)

## Unanswerable (~20)

### U-001
- **Question:** In which city is Brightwater Studio's office?
- **Why it can't be answered:** no document gives the studio's address or city
- **Notes:** v0 (Claude-authored, provisional)

### U-002
- **Question:** Which make and model is the graphics card in the Lantern computer?
- **Why it can't be answered:** the hardware notes give only its 12 GB of video memory, never a make or model
- **Notes:** v0 (Claude-authored, provisional)

### U-003
- **Question:** What is the name of the language model that Lantern runs?
- **Why it can't be answered:** the documents describe the model by size and precision only (8 billion parameters, 4-bit), never by name
- **Notes:** v0 (Claude-authored, provisional)

### U-004
- **Question:** What topics does the 45 minute training session cover, in order?
- **Why it can't be answered:** the documents give its length and dates, but no agenda
- **Notes:** v0 (Claude-authored, provisional)

### U-005
- **Question:** How many prompts did the whole studio send to Lantern in June?
- **Why it can't be answered:** only the four-week pilot has usage figures; none are given for June
- **Notes:** v0 (Claude-authored, provisional)

### U-006
- **Question:** What satisfaction score did all 12 staff give Lantern after the full rollout?
- **Why it can't be answered:** only the pilot survey (four people) has a result; no survey after rollout is reported
- **Notes:** v0 (Claude-authored, provisional)

### U-007
- **Question:** What is Maya Okafor's email address?
- **Why it can't be answered:** the runbook gives phone numbers only, no email addresses
- **Notes:** v0 (Claude-authored, provisional)

### U-008
- **Question:** What is Priya Nair's daily rate?
- **Why it can't be answered:** the documents that chat mode reads state a monthly support retainer, never a daily rate (the workbook's rows are not indexed in chat mode)
- **Notes:** v0 (Claude-authored, provisional)

### U-009
- **Question:** In which cities do Harbour and Pine and Corvid Foods have their offices?
- **Why it can't be answered:** the clients are named, but nothing is said about where they are
- **Notes:** v0 (Claude-authored, provisional)

### U-010
- **Question:** What penalty do the client NDAs set if their data is leaked?
- **Why it can't be answered:** the documents say the NDAs forbid third-party processing, not what the penalties are
- **Notes:** v0 (Claude-authored, provisional)

### U-011
- **Question:** Can staff talk to Lantern by voice?
- **Why it can't be answered:** no document mentions voice input or output
- **Notes:** v0 (Claude-authored, provisional)

### U-012
- **Question:** How many of the 12 staff work part-time?
- **Why it can't be answered:** the headcount of 12 is given, but not how the staff work
- **Notes:** v0 (Claude-authored, provisional)

### U-013
- **Question:** Which questions were asked in the pilot survey?
- **Why it can't be answered:** only the survey's result (4.3 out of 5) is reported, never its questions
- **Notes:** v0 (Claude-authored, provisional)

### U-014
- **Question:** What is the minimum password length for Lantern accounts?
- **Why it can't be answered:** the documents mention individual accounts and multi-factor sign-in for administrators, but no password rules
- **Notes:** v0 (Claude-authored, provisional)

### U-015
- **Question:** How many tokens per second would a model with 70 billion parameters reach on the Lantern computer?
- **Why it can't be answered:** only the 8 billion parameter model was measured
- **Notes:** v0 (Claude-authored, provisional)

### U-016
- **Question:** Who takes over as security and privacy officer if Tomasz Lewandowski leaves?
- **Why it can't be answered:** no document names a deputy or a succession plan
- **Notes:** v0 (Claude-authored, provisional)

### U-017
- **Question:** Which operating system is installed on lantern-01?
- **Why it can't be answered:** the hardware and security documents never name the operating system
- **Notes:** v0 (Claude-authored, provisional)

### U-018
- **Question:** How many watts does the Lantern computer draw under load?
- **Why it can't be answered:** no document gives its power consumption, only the battery run-time
- **Notes:** v0 (Claude-authored, provisional)

### U-019
- **Question:** What was the studio's revenue last year?
- **Why it can't be answered:** the documents cover the AI rollout only and say nothing about the studio's finances beyond its costs
- **Notes:** v0 (Claude-authored, provisional)

### U-020
- **Question:** How many people attended the June refresher session?
- **Why it can't be answered:** the June meeting only plans the refresher; no document reports who attended it
- **Notes:** v0 (Claude-authored, provisional)
