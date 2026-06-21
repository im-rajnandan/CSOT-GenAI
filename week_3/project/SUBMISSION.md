# Week 3 Project Submission

For Week 3, I upgraded my Week 2 ResearchBot into **Research Desk**. The Week 2
version worked, but almost everything lived in one large Python file and every
conversation disappeared when the program closed. This version separates the agent
brain from the terminal interfaces and saves each conversation as JSON, so I can
resume the same research later.

The main `Agent` class owns the model loop, tool registry, session history, and tool
dispatch. `REPLAgent` only handles terminal input and output, while `TUIAgent` sends
the same agent events to a Textual application. I kept streaming from my Week 2
project, including the part that joins tool names and JSON arguments when OpenRouter
sends them across multiple chunks. The TUI now has a separate activity panel, which
keeps tool calls and errors out of the actual conversation.

I replaced the AlphaXiv MCP process with two direct Hugging Face tools:
`paper_search` and `read_paper`. Search returns small results, and reading a paper
tries the available Markdown before falling back to its abstract. Hugging Face does
not contain every arXiv paper, so the tool returns an arXiv fallback URL when needed.
The old Serper search and readable-page extraction still use `trafilatura` first and
`markdownify` as a fallback.

Another major change is workspace access. The agent can list and read files, create
research notes, and make line-based replace, delete, or append edits. Every path is
resolved inside `WORKSPACE_ROOT`, and edits return a unified diff so the change is
visible to the model. Project behavior such as citation style and when to save notes
lives in `AGENTS.md`, instead of being buried entirely inside the Python prompt.

The most important design decision was keeping one agent loop for every interface.
It is tempting to put separate API logic inside the TUI, but that caused debugging
friction in Week 2. Now one-shot mode, the REPL, and the TUI all use the same session
and tool behavior. If I had more time, I would add context compaction for very long
sessions, because saving full tool results is correct for resuming but will eventually
fill the model context window.
