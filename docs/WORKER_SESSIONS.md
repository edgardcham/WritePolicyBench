# Worker sessions

We use two long-running CLI "workers" that Loopy coordinates:

## Codex worker (implementation)
- Start: `codex`
- Resume: `codex resume --last`
- Non-interactive: `codex exec "<task>"`

## Claude worker (research + writing)
- Start: `claude`
- Non-interactive: `claude -p "<task>" --output-format text`

## Convention
When starting a worker on a task:
- record: date, goal, commands used, and how to resume
- store in Notion Tasks entry under "Worker" section
