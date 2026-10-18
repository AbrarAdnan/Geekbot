# Lantern incident runbook

Owner: Tomasz Lewandowski, security and privacy officer. Version 1.2, 30 April 2025.

Use this when something has gone wrong with Lantern, or with any AI tool at Brightwater Studio, and client or staff data may be affected.

## Who to call

| Person | Role | Phone |
|---|---|---|
| Tomasz Lewandowski | Security and privacy officer, runs every incident | 07700 900101 |
| Priya Nair | IT contractor, on call on Mondays and Thursdays | 07700 900102 |
| Maya Okafor | Managing director, speaks to clients about serious incidents | 07700 900103 |

If you cannot reach Tomasz, call Maya.

## When to report

The policy sets the deadlines. Any suspected incident must be reported to Tomasz within 24 hours. If Red data may have been exposed, call him by phone within 1 hour, and do not wait to send an email.

## How serious is it?

| Level | What it means | Acknowledge | Contain | Tell the client |
|---|---|---|---|---|
| S1 | Red data exposed, or Lantern compromised | within 15 minutes | within 1 hour | within 24 hours |
| S2 | Amber data exposed, or a wrong output reached a client | within 4 hours | the same working day | within 72 hours, if their data was involved |
| S3 | An outage or a small breach of the rules with no data exposed | the next working day | within 3 working days | no notice needed |

An S3 incident is discussed at the weekly project meeting. S1 and S2 incidents get a review meeting.

## What to do, in order

1. **Stop it.** Priya or Tomasz disables the affected account in the admin console. If a tool outside Lantern is involved, stop using it.
2. **Record it.** Add a row to the incident log (a CSV file in the ops folder) with the next INC number, the date, the level, who reported it and a one-line summary.
3. **Preserve the evidence.** Create an empty file called LEGAL_HOLD in /srv/lantern. While it exists the nightly retention job deletes nothing, so the logs you need cannot expire.
4. **Tell the right people.** Tomasz decides who must be told. Maya speaks to the client. Nobody else contacts a client about an incident.
5. **Fix it.** Priya or the owner of the affected system makes the repair and notes what was changed.
6. **Review it.** Tomasz holds a review within 5 working days of the incident being contained. Only after he signs off is the LEGAL_HOLD file removed.

## Records

Incident records are kept for three years. The log shows the status of each incident: open, resolved or closed. An incident is closed only after its review.
