"""Textual interface for the persistent Research Desk agent."""

from __future__ import annotations

import json
from typing import Any, Callable

from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Input, RichLog, Static

from agent import Agent, QUIT_COMMAND


class TUIAgent(Agent):
    """Agent presentation adapter that forwards events to a Textual app."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.event_sink: Callable[[str, dict[str, Any]], None] | None = None

    def _emit(self, event: str, **data: Any) -> None:
        if self.event_sink is not None:
            self.event_sink(event, data)

    def run(self) -> None:
        ResearchDeskApp(self).run()


class ChatBubble(Static):
    """A Markdown-rendered user or assistant message."""

    def __init__(self, role: str, content: str = "", streaming: bool = False):
        super().__init__(classes=f"bubble {role}")
        self.role = role
        self.content = content
        self.streaming = streaming

    def add_text(self, text: str) -> None:
        self.content += text
        self.streaming = True
        self.refresh(layout=True)

    def finish(self) -> None:
        self.streaming = False
        self.refresh(layout=True)

    def render(self) -> Group:
        title = "You" if self.role == "user" else "Research Desk"
        style = "bold green" if self.role == "user" else "bold cyan"
        body = self.content or ("Thinking…" if self.streaming else "")
        return Group(Text(title, style=style), Markdown(body))


class ResearchDeskApp(App):
    """Split-panel chat and tool-activity interface."""

    TITLE = "Research Desk"
    BINDINGS = [
        Binding("ctrl+l", "clear_display", "Clear display", priority=True),
        Binding("ctrl+k", "new_session", "New session", priority=True),
        Binding("ctrl+q", "quit_app", "Quit", priority=True),
        Binding("ctrl+u", "clear_input", "Clear input", priority=True),
    ]

    CSS = """
    Screen {
        layout: vertical;
        background: #0d1117;
        color: #c9d1d9;
    }
    #top {
        height: 1;
        margin: 1 2 0 2;
        color: #e6edf3;
        text-style: bold;
    }
    #main {
        height: 1fr;
        margin: 1 2 0 2;
    }
    #chat {
        width: 3fr;
        padding: 1 2;
        border: round #30363d;
    }
    #tools {
        width: 1fr;
        margin-left: 1;
        padding: 1;
        border: round #30363d;
        background: #161b22;
    }
    .bubble {
        height: auto;
        margin: 0 0 2 0;
    }
    #status {
        height: 1;
        margin: 0 3;
        color: #8b949e;
    }
    #input {
        height: 3;
        margin: 0 2 1 2;
        border: round #30363d;
        background: #161b22;
    }
    #input:focus {
        border: round #58a6ff;
    }
    """

    def __init__(self, agent: TUIAgent):
        super().__init__()
        self.agent = agent
        self.busy = False
        self.current_bot_bubble: ChatBubble | None = None

    def compose(self) -> ComposeResult:
        yield Static("", id="top")
        with Horizontal(id="main"):
            yield VerticalScroll(id="chat")
            yield RichLog(id="tools", wrap=True, markup=True, highlight=True)
        yield Static("Ready", id="status")
        yield Input(placeholder="Ask a question or type /help…", id="input")

    def on_mount(self) -> None:
        self.agent.event_sink = lambda event, data: self.call_from_thread(
            self.apply_agent_event, event, data
        )
        self._update_header()
        self.rebuild_chat()
        self.query_one("#tools", RichLog).write("[bold]Tool activity[/bold]")
        self.query_one("#input", Input).focus()

    def _update_header(self) -> None:
        self.query_one("#top", Static).update(
            f"Research Desk — {self.agent.session_id}: {self.agent.title}  "
            "| Ctrl+L clear | Ctrl+K new | Ctrl+Q quit"
        )

    def set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def add_bubble(self, role: str, content: str = "", streaming: bool = False) -> ChatBubble:
        bubble = ChatBubble(role, content, streaming=streaming)
        chat = self.query_one("#chat", VerticalScroll)
        chat.mount(bubble)
        chat.scroll_end(immediate=True)
        return bubble

    def rebuild_chat(self) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        chat.remove_children()
        self.current_bot_bubble = None
        for message in self.agent.visible_messages():
            self.add_bubble(message["role"], message["content"])
        self._update_header()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        event.input.clear()
        if not question or self.busy:
            return
        self._launch(question, show_user=not question.startswith("/"))

    def _launch(self, question: str, *, show_user: bool) -> None:
        if self.busy:
            return
        self.busy = True
        input_box = self.query_one("#input", Input)
        input_box.disabled = True
        if show_user:
            self.add_bubble("user", question)
        self.current_bot_bubble = None
        self.set_status("Working…")
        self.run_worker(lambda: self.worker(question), thread=True, exclusive=True)

    def worker(self, question: str) -> None:
        try:
            if question.startswith("/"):
                result = self.agent.handle_command(question)
                if result == QUIT_COMMAND:
                    self.agent.save_session()
                    self.call_from_thread(self.exit)
                    return
                self.call_from_thread(self.finish_local_result, result or "")
                return
            answer = self.agent.chat(question)
            self.call_from_thread(self.finish_answer, answer)
        except Exception as exc:
            self.call_from_thread(self.finish_error, str(exc))

    def finish_local_result(self, result: str) -> None:
        if result:
            self.add_bubble("assistant", result)
        self._finish_worker()

    def finish_answer(self, answer: str) -> None:
        if self.current_bot_bubble is not None:
            streamed = self.current_bot_bubble.content
            self.current_bot_bubble.finish()
            if answer and answer != streamed:
                self.add_bubble("assistant", answer)
        elif answer:
            self.add_bubble("assistant", answer)
        self.current_bot_bubble = None
        self._update_header()
        self._finish_worker()

    def finish_error(self, message: str) -> None:
        self.query_one("#tools", RichLog).write(f"[bold red]Error:[/bold red] {message}")
        self.add_bubble("assistant", f"Request failed: {message}")
        self._finish_worker()

    def _finish_worker(self) -> None:
        self.busy = False
        input_box = self.query_one("#input", Input)
        input_box.disabled = False
        input_box.focus()
        self.set_status("Ready")

    def apply_agent_event(self, event: str, data: dict[str, Any]) -> None:
        tools = self.query_one("#tools", RichLog)
        if event == "step_started":
            if self.current_bot_bubble is not None:
                self.current_bot_bubble.finish()
                self.current_bot_bubble = None
            self.set_status(f"Thinking — step {data.get('step')}/{data.get('maximum')}")
        elif event == "token":
            if self.current_bot_bubble is None:
                self.current_bot_bubble = self.add_bubble("assistant", streaming=True)
            self.current_bot_bubble.add_text(str(data.get("text", "")))
            self.query_one("#chat", VerticalScroll).scroll_end(immediate=True)
        elif event == "tool_started":
            preview = json.dumps(data.get("arguments", {}), ensure_ascii=False, default=str)
            if len(preview) > 140:
                preview = preview[:137] + "…"
            tools.write(f"[yellow]→ {data.get('name')}[/yellow]\n  {preview}")
            self.set_status(f"Using {data.get('name')}…")
        elif event == "tool_finished":
            tools.write(f"[green]✓ {data.get('name')}[/green]")
        elif event == "tool_error":
            result = data.get("result", {})
            tools.write(f"[bold red]✗ {data.get('name')}[/bold red]\n  {result.get('error', '')}")
        elif event == "session_changed":
            tools.write(
                f"[cyan]Session {data.get('action')}:[/cyan] {data.get('session_id')}"
            )
            self.rebuild_chat()
        elif event == "completed":
            self.set_status("Done")
        elif event == "error":
            tools.write(f"[bold red]Error:[/bold red] {data.get('message', '')}")

    def action_clear_display(self) -> None:
        self.query_one("#chat", VerticalScroll).remove_children()
        self.query_one("#tools", RichLog).clear()
        self.current_bot_bubble = None
        self.set_status("Display cleared; session history retained")

    def action_new_session(self) -> None:
        if not self.busy:
            self._launch("/new", show_user=False)

    def action_quit_app(self) -> None:
        self.agent.save_session()
        self.exit()

    def action_clear_input(self) -> None:
        self.query_one("#input", Input).clear()
        self.set_status("Input cleared")
