# Security Policy

## Reporting

Please report vulnerabilities privately through GitHub Security Advisories for
`qiu01shi/pharos`. Do not include credentials or sensitive trace data in a
public issue.

## Security boundary

Pharos permissions are capability authorization, not OS-level isolation.
Granting shell or filesystem capabilities allows the corresponding entity or
tool to act with the privileges of the pharos process. Run untrusted agents in
a container or another sandbox, restrict network access, and reference secrets
from the environment instead of embedding them in graph files.

Run records and traces may contain prompts, tool arguments, and outputs. Treat
the `.pharos` data directory as sensitive until configurable redaction and
encrypted storage are available.
