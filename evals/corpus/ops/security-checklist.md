# Lantern pre-launch security checklist

Completed on 9 May 2025 and signed off by Tomasz Lewandowski. Full rollout to all staff followed on 12 May 2025.

Each check below was carried out by the person named, on the date given.

| Check | Result | Done by | Date |
|---|---|---|---|
| Full-disk encryption on lantern-01 | Enabled | Priya Nair | 29 March |
| Network access | Only the office subnet 192.168.10.0/24 can reach Lantern; no port is open to the internet | Priya Nair | 30 March |
| Administrator accounts | Two administrators, both with multi-factor sign-in | Priya Nair | 31 March |
| Retention job | Test files aged 91 days were deleted at the next run; files aged 89 days were kept | Priya Nair | 30 April |
| Legal hold | With the LEGAL_HOLD file present the job deleted nothing and exited with code 3 | Priya Nair | 30 April |
| Power backup | The workstation ran for 21 minutes on battery before the test was stopped | Priya Nair | 28 April |
| Backup restore | A full restore of 310 GB took 47 minutes | Priya Nair | 2 May |
| User accounts | 12 individual accounts and no shared accounts | Daniel Reyes | 9 May |
| Training | 12 of 12 staff attended a training session, on 5 and 6 May | Daniel Reyes | 6 May |
| Policy | The AI use policy was signed by Maya Okafor | Maya Okafor | 21 March |

## Open points at sign-off

- The 95th percentile response time was inside the 3 second target after the change of 14 April, but only just (2.8 seconds). Priya will repeat the measurement after eight weeks of full use.
- The incident log and the runbook are kept by Tomasz. A second person should learn the runbook, so Daniel Reyes will attend the first review meeting.
