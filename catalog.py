"""Phase-1 tool catalog: a realistic Claude Code-style toolset.

Descriptions are the routing criteria Jev sees. Deliberately overlapping
pairs (Read/Grep/Glob, Bash vs dedicated tools, WebFetch/WebSearch,
Grep vs LSP-references) keep routing non-trivial.
"""

CATALOG: dict[str, str] = {
    "Read": "Read the contents of a specific known file by path",
    "Write": "Create a new file or fully replace a file's contents",
    "Edit": "Make an exact in-place string replacement inside an existing file",
    "Bash": "Run a shell command: git, tests, linters, installs, scripts",
    "Grep": "Search file CONTENTS for a regex/text pattern across the codebase",
    "Glob": "Find files by NAME/path pattern, e.g. tests/**/*_test.py",
    "ListDir": "List the files and subdirectories of one directory",
    "WebFetch": "Fetch a specific known URL and extract information from it",
    "WebSearch": "Search the web when no specific URL is known",
    "Agent": "Spawn a subagent for a large independent subtask or parallel work",
    "TodoWrite": "Create or update the tracked todo/plan list for the task",
    "NotebookEdit": "Edit, insert, or delete a cell in a Jupyter notebook",
    "AskUserQuestion": "Ask the user a clarifying question and wait for the answer",
    "TaskStop": "Stop or kill a running background task/shell by id",
    "LSP": "Code intelligence: go-to-definition, find-references, symbol types",
}
