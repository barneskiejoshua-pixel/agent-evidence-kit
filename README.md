# agent-evidence-kit

Eight single-file tools from a local agent whose rule is that nothing counts as done because the
model said so. Each file stands alone and uses only the standard library.

| file | what it settles |
| --- | --- |
| `calc.py` | arithmetic is computed exactly (decimal, not float) and never by the model |
| `tabular.py` | a number read out of a CSV or xlsx is computed by sqlite; only one SELECT is allowed |
| `redact.py` | secrets are replaced before tool output reaches the model or the screen |
| `untrusted.py` | text from files and web pages is handed to the model as data, with injected orders flagged |
| `patching.py` | line edits and unified diffs that keep the file's encoding and line endings, written atomically |
| `journal.py` | what the file system says changed while a command ran (NTFS USN journal on Windows) |
| `integrity.py` | a green test run counts only if it ran the program the task named |
| `grounding.py` | an answer is checked sentence by sentence against the facts it was given |

    python check.py

`journal.py` is Windows-specific. `integrity.py` expects the host agent's trace format; read its
docstring before reusing it.
