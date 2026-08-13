# Platform Engineering Runbook

## Deploys

Deploys run from the `main` branch only. A deploy is gated on the full test
suite plus a smoke check against staging; either failing blocks promotion.

Rollbacks are a one-liner: `plat rollback <service> --to <release-id>`. The
previous three releases stay warm, so a rollback completes in under 40
seconds. Anything older requires a rebuild.

Deploy windows are 09:00–16:00 IST on weekdays. Outside that window a deploy
requires an override from an on-call engineer, recorded in the change log.

## Incident response

Severity one means customer-facing data loss or a total outage. Page the
on-call immediately and open a war room channel. Severity two is degraded
service with a workaround; it waits for business hours.

The first responder owns communication until an incident commander takes over
explicitly. Do not assume someone else is writing updates — silence during an
incident is itself a failure mode.

Every severity-one incident gets a written postmortem within five business
days. Postmortems are blameless and are published to the whole engineering
org, not just the team involved.

## Database migrations

Migrations must be backwards compatible with the previously deployed release.
That means: add columns nullable, backfill in a separate job, and only drop a
column one release after the last reader is gone.

Never run a migration inside a request handler. Long-running `ALTER TABLE` on
Postgres takes an ACCESS EXCLUSIVE lock and will queue every subsequent query
against that table until it completes.

The migration runner refuses to apply a migration that has no down-migration,
unless it is explicitly marked irreversible with a written justification.

## On-call

On-call rotations are one week, handing over at 10:00 on Wednesdays. The
outgoing engineer walks the incoming one through anything still open.

Compensation for on-call is a flat weekly stipend plus time-in-lieu for any
page received between 22:00 and 07:00. Time-in-lieu must be taken within the
following month.

If you are paged twice in one night, escalate to the secondary rather than
continuing alone. Fatigue-driven mistakes during incidents are the most common
cause of a severity-one becoming a severity-zero.
