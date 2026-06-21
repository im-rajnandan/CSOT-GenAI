# CSOT GenAI / Agentic Track

This repository contains my weekly projects for the CSOT 2026 GenAI/Agentic track.

## Week 2 — ResearchBot

The original Week 2 submission is preserved in [`week_2/project/`](week_2/project/). It
implements a streaming web-and-paper research agent with a Textual interface and
AlphaXiv MCP integration.

## Week 3 — Research Desk

Week 3 lives in [`week_3/project/`](week_3/project/). It refactors the agent into a
reusable class hierarchy and adds:

- persistent JSON sessions and resumable conversations;
- project rules loaded from `AGENTS.md`;
- direct Hugging Face paper search and reading;
- sandboxed read, write, list, and line-edit tools;
- durable Markdown research notes;
- one-shot, REPL, and split-panel Textual interfaces.

### Setup

```bash
cd week_3/project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `OPENROUTER_API_KEY` and `SERPER_API_KEY`. `HF_TOKEN` is optional.

### Run

```bash
python agent.py
python agent.py "Summarise the FlashAttention paper"
python agent.py --tui
python agent.py --session a3f8c2d1
python agent.py --list-sessions
```

Inside the REPL or TUI, use `/sessions`, `/resume <id>`, `/new`, `/help`, and
`/quit`. The TUI also supports `Ctrl+L`, `Ctrl+K`, `Ctrl+U`, and `Ctrl+Q`.

Hugging Face Papers indexes a large ML/CS subset of arXiv, not every paper. When a
paper is missing, Research Desk falls back to the corresponding arXiv web page.
