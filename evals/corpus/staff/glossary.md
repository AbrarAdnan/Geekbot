# Lantern glossary

Short definitions of the words you will meet in the Lantern documents at Brightwater Studio.

**Amber data.** Client work in progress and internal plans. Lantern may process it. Outside AI tools may not.

**Audit log.** A record of who used Lantern, when, in which workspace and how many tokens were used. It never contains the text of a prompt or an answer. Kept for 365 days.

**Cleanup job.** The script log_cleanup.py that runs on lantern-01 every night at 02:30 and deletes chat logs and audit logs that have passed their retention period.

**Embedder.** The small model that turns a document into numbers so that Lantern can search it by meaning. It runs on the processor, not on the graphics card.

**First-token time.** How long you wait before the first word of an answer appears. The studio's target is under 3 seconds for 95 percent of requests.

**Green data.** Published or public material, such as marketing copy that is already on the website. Anyone may use it with any tool.

**Lantern.** The private AI assistant that runs on the workstation lantern-01. It is the result of Project Lantern.

**Legal hold.** An empty file called LEGAL_HOLD in /srv/lantern. While it exists the cleanup job deletes nothing. Used during an incident so that evidence is kept.

**NDA.** A non-disclosure agreement. Two clients, Harbour and Pine and Corvid Foods, have NDAs that do not allow third parties to process their work.

**Pilot group.** The four people who tested Lantern from 7 April to 2 May 2025: Hannah Frost, Daniel Reyes, Samir Haddad and Leo Park.

**Quantised.** A model that has been stored with fewer bits per number so that it takes less memory. Lantern's model uses 4 bits and needs about 5 GB.

**Red data.** Client contracts, personal data and financial records. It may go into Lantern only with written approval from Tomasz Lewandowski.

**Retention.** How long something is kept before it is deleted. Chat logs: 90 days.

**Severity levels.** S1 is the most serious incident (Red data exposed or Lantern compromised), S2 is Amber data exposed or a wrong output reaching a client, S3 is an outage or a small breach with no data exposed.

**Token.** A piece of a word. Models read and write in tokens; about four characters make one token in English.

**Workspace.** A separate folder of documents and conversations for one client project. A document uploaded to one workspace cannot be seen from another.
