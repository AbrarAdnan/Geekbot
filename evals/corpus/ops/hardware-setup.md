# Lantern hardware and performance

Last updated 2 July 2025 by Priya Nair. Applies to the Lantern assistant at Brightwater Studio.

## The machine

Lantern runs on a single workstation called lantern-01. It lives in the locked comms cupboard on the ground floor, on its own shelf with its own power socket, so nobody needs to touch it during normal work.

| Part | Specification |
|---|---|
| Graphics card | 12 GB of video memory |
| Memory | 64 GB of RAM |
| Storage | 2 TB NVMe SSD, fully encrypted |
| Processor | 8 cores |
| Power backup | UPS (battery unit), bought for £180 |

The workstation itself cost £2,150. It was ordered on 17 March 2025 and delivered on 27 March. Priya Nair set it up between 28 and 31 March.

## The model

The assistant is an open-weight language model with 8 billion parameters. It is stored in 4-bit precision (the term is "quantised"), which brings it down to about 5 GB of video memory. The separate model that turns documents into searchable vectors, the embedder, runs on the processor instead, so the graphics card is left free for the chat model.

## Speed

Priya measured the speed on 31 March 2025, before the pilot:

- The median time to the first word of an answer was 1.8 seconds.
- The 95th percentile was 3.4 seconds. The target in the vendor memo had been under 3 seconds at the 95th percentile, so the first benchmark missed it.
- Text came out at 38 tokens per second.

On 14 April Priya shortened the context length to 4,096 tokens and measured again. The 95th percentile dropped to 2.8 seconds, inside the target.

## How many people at once

Up to 3 people can chat at the same time without a noticeable slowdown. A fourth request waits in a queue until one of the others finishes. For a studio of 12 this has been enough so far, because people use the assistant in short bursts.

## Power

The UPS is rated to keep the workstation running for about 20 minutes at normal load. On 1 July 2025 a power cut lasted 40 minutes. The UPS held for 22 minutes and then the workstation shut itself down cleanly, with no data lost (incident INC-008).

## Scheduled jobs

| When | What |
|---|---|
| Every night at 01:00 | Backup to the office NAS |
| Every night at 02:30 | Retention job that deletes expired logs (the script is log_cleanup.py) |
| Every Sunday at 03:00 | Encrypted copy of the backup to cloud storage in an EU region |

Backups are kept for 30 days. A file that the retention job deleted can therefore still sit in a backup for up to 30 days afterwards.
