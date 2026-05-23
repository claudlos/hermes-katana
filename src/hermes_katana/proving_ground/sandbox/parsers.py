"""Model-specific output parsers for pseudo tool call extraction.

Different models format their outputs differently:
- Qwen: Uses ```bash code blocks, $ prefixed commands
- Gemma: Uses <code> tags, plain command text, function call syntax
- Bonsai: Varies (1-bit models are less structured)
- Nemotron / TinyLlama: Often write prose ("you should remove the file")
"""

import re


# Models leak their stop tokens into content; strip before parsing.
STOP_TOKEN_RE = re.compile(r"<\|(?:im_end|im_start|endoftext|eot_id|start_header_id|end_header_id|begin_of_text)\|>")


def _strip_stop_tokens(text: str) -> str:
    return STOP_TOKEN_RE.sub("", text)


# Verbs we recognise as introducing a shell-style command in prose.
_CMD_VERB_GROUP = r"(?:run|running|execute|executing|type|typing|use|using|invoke|invoking|try)"
_INTENT_PREFIX = r"(?:you\s+can|you\s+should|we\s+can|we\s+should|let'?s|let\s+us|i'?ll|i\s+will)\s+"
# Bash command head we'll recognise after a verb. Word-boundary anchored.
_BASH_HEAD = (
    # Longer multi-word commands first so the alternation prefers them.
    r"(?:docker-compose|python3|"
    r"cat|ls|find|grep|head|tail|cd|pwd|mkdir|rm|cp|mv|chmod|chown|"
    r"python|node|npm|git|curl|wget|docker|make|"
    r"awk|sed|tar|sh|bash|kill|ps|systemctl|service)"
)
# Up to ~3 path-like or flag tokens following the head. Allows internal dots
# (so src/app.py stays whole) but stops at sentence-ending punctuation.
_CMD_TOKEN = r'[^\s`"\',;:!?\n][^\s`"\',;:!?\n]*'
_CMD_TAIL = rf"(?:\s+{_CMD_TOKEN}){{0,3}}"

_PROSE_PATTERNS = [
    # "run/execute/use [the] [command] X", "type ls", "running cat foo"
    (
        re.compile(
            rf"\b{_CMD_VERB_GROUP}\s+(?:the\s+)?(?:command\s+)?"
            rf"(?:`([^`]{{1,80}})`"
            rf'|"([^"]{{1,80}})"'
            rf"|'([^']{{1,80}})'"
            rf"|({_BASH_HEAD}\b{_CMD_TAIL}))",
            re.IGNORECASE,
        ),
        "cmd",
    ),
    # "remove/delete/read/inspect the file X" — verb + path-shaped target
    (
        re.compile(
            r"\b(remove|delete|read|view|open|list|inspect|check|examine)\s+"
            r"(?:the\s+)?(?:file\s+|directory\s+|folder\s+|contents?\s+of\s+)?"
            r"(?:`([^`]{1,80})`"
            r'|"([^"]{1,80})"'
            r"|'([^']{1,80})'"
            r"|([\w./\-]+\.[a-zA-Z]{1,5})"
            r"|([\w./\-]+/[\w./\-]+))",
            re.IGNORECASE,
        ),
        "verb_target",
    ),
    # "you can run X", "we should execute Y"
    (
        re.compile(
            rf"\b{_INTENT_PREFIX}{_CMD_VERB_GROUP}\s+"
            rf"(?:`([^`]{{1,80}})`"
            rf'|"([^"]{{1,80}})"'
            rf"|'([^']{{1,80}})'"
            rf"|({_BASH_HEAD}\b{_CMD_TAIL}))",
            re.IGNORECASE,
        ),
        "cmd",
    ),
]


_NOISE_WORDS = {
    "command",
    "commands",
    "tool",
    "tools",
    "shell",
    "terminal",
    "option",
    "options",
    "flag",
    "flags",
    "argument",
    "arguments",
}
# Trailing English stopwords that signal the command part has ended in prose.
_STOPWORDS = {
    "to",
    "and",
    "or",
    "but",
    "so",
    "then",
    "for",
    "with",
    "from",
    "on",
    "in",
    "at",
    "by",
    "of",
    "a",
    "an",
    "the",
    "this",
    "that",
    "see",
    "look",
    "check",
    "find",
    "use",
    "run",
    "first",
    "next",
    "is",
    "was",
    "be",
    "will",
    "would",
    "should",
    "can",
    "could",
    "do",
    "does",
    "did",
    "when",
    "while",
    "because",
    "if",
}


def _classify_cmd(cmd: str) -> tuple[str, dict] | None:
    """Map a shell command string to a (tool, args) tuple. Returns None if noise."""
    cmd = cmd.strip().strip("`\"' ")
    if not cmd:
        return None
    parts = cmd.split()
    # Keep head + arg-like tokens. Stop at the first English stopword or noise word.
    head = parts[0]
    cleaned = [head]
    for token in parts[1:]:
        low = token.lower().rstrip(".,;:!?")
        if low in _STOPWORDS or low in _NOISE_WORDS:
            break
        cleaned.append(token.rstrip(".,;:!?"))
    parts = cleaned
    if not parts:
        return None
    cmd = " ".join(parts)
    head = parts[0]
    # A bare command verb with no target is too vague
    if len(parts) == 1 and head in (
        "cat",
        "head",
        "tail",
        "rm",
        "cp",
        "mv",
        "grep",
        "find",
        "echo",
    ):
        return None
    if head in ("ls", "dir", "tree"):
        return ("list_files", {"path": "."})
    if head == "find":
        # find is too ambiguous in prose ("find the file"); treat as a generic command
        return ("run_command", {"command": cmd, "timeout": 30})
    if head in ("cat", "head", "tail", "less", "more", "view") and len(parts) >= 2:
        target = parts[-1].strip("`\"' ")
        # Reject obvious noise targets ("the", "file" etc.)
        if target.lower() in _NOISE_WORDS or target.lower() in {"the", "a", "an"}:
            return None
        return ("read_file", {"path": target})
    if ">" in cmd and head == "echo":
        before, after = cmd.split(">", 1)
        return (
            "write_file",
            {
                "path": after.strip().strip("`\"' "),
                "content": before.replace("echo", "", 1).strip().strip("`\"' "),
            },
        )
    return ("run_command", {"command": cmd, "timeout": 30})


def _classify_verb_target(verb: str, target: str) -> tuple[str, dict] | None:
    """Map a (verb, target) prose pair to a tool call."""
    verb = verb.lower()
    target = target.strip().strip("`\"' ")
    if not target:
        return None
    # Skip generic targets that aren't real things
    generic = {
        "file",
        "files",
        "directory",
        "directories",
        "folder",
        "folders",
        "it",
        "this",
        "that",
        "them",
        "these",
        "those",
        "one",
        "output",
        "content",
        "contents",
        "data",
        "value",
        "values",
        "something",
        "anything",
        "everything",
        "nothing",
    }
    if target.lower() in generic or target.lower() in _NOISE_WORDS:
        return None
    # Reject targets that don't look like real file/path identifiers
    looks_like_path = "/" in target or ("." in target and len(target.split(".")[-1]) <= 5)
    if verb in ("read", "view", "open", "inspect", "examine", "check"):
        if looks_like_path:
            return ("read_file", {"path": target})
        return None
    if verb == "list":
        return ("list_files", {"path": target if "/" in target else "."})
    if verb in ("remove", "delete"):
        if looks_like_path:
            return ("run_command", {"command": f"rm {target}", "timeout": 30})
        return None
    return None


def _extract_prose_commands(text: str) -> list[tuple[str, dict]]:
    """Extract commands from natural-language prose."""
    out: list[tuple[str, dict]] = []
    for pattern, kind in _PROSE_PATTERNS:
        for match in pattern.finditer(text):
            groups = [g for g in match.groups() if g]
            if not groups:
                continue
            if kind == "cmd":
                cmd = groups[-1].strip()
                tc = _classify_cmd(cmd) if cmd else None
                if tc:
                    out.append(tc)
            elif kind == "verb_target":
                verb = match.group(1)
                target = next((g for g in match.groups()[1:] if g), "")
                tc = _classify_verb_target(verb, target)
                if tc:
                    out.append(tc)
    return out


def _dedupe_commands(cmds: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """Drop exact duplicates while preserving order."""
    seen: set = set()
    out: list[tuple[str, dict]] = []
    for tool, args in cmds:
        # Normalise args to a hashable representation
        key = (tool, tuple(sorted((k, str(v)) for k, v in args.items())))
        if key in seen:
            continue
        seen.add(key)
        out.append((tool, args))
    return out


class BaseParser:
    """Base class for model-specific output parsers."""

    # Subclasses can disable prose extraction if it produces too many false positives.
    use_prose_extractor: bool = True

    def extract_commands(self, text: str) -> list[tuple[str, dict]]:
        """Extract tool-like commands from model output."""
        return []

    def _augment_with_prose(self, text: str, existing: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
        """Add prose-derived commands and dedupe against existing structured matches."""
        if not self.use_prose_extractor:
            return _dedupe_commands(existing)
        prose = _extract_prose_commands(text)
        return _dedupe_commands(existing + prose)


class QwenParser(BaseParser):
    """Parser for Qwen-family models (Qwen3, Qwen3.5).

    Qwen models typically write:
    - ```bash code blocks
    - $ prefixed commands
    - Direct file references like "cat src/app.py"
    """

    CODE_BLOCK = re.compile(r"```(?:bash|sh|shell)?\s*\n(.+?)\n```", re.DOTALL)
    SINGLE_CMD = re.compile(r"^(?:\$|#|>)\s*(.+)$", re.MULTILINE)
    FILE_REF = re.compile(r"(?:cat|read|open|view)\s+(\S+\.\w+)", re.IGNORECASE)
    # Prose patterns: "use the `find` command", "run `ls -la`", "using ls"
    INLINE_CMD = re.compile(r"(?:use|run|execute|type|using)\s+(?:the\s+)?`([^`]+)`", re.IGNORECASE)
    BACKTICK_CMD = re.compile(
        r"`((?:ls|cat|find|grep|head|tail|cd|mkdir|rm|cp|mv|chmod|python|git|curl|wget|echo)\s+[^`]+)`"
    )

    def extract_commands(self, text: str) -> list[tuple[str, dict]]:
        text = _strip_stop_tokens(text)
        commands = []

        # Code blocks
        for match in self.CODE_BLOCK.finditer(text):
            block = match.group(1).strip()
            for line in block.split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("echo"):
                    commands.append(self._parse_cmd(line))

        # Single-line commands
        for match in self.SINGLE_CMD.finditer(text):
            cmd = match.group(1).strip()
            if cmd:
                commands.append(self._parse_cmd(cmd))

        # File references
        for match in self.FILE_REF.finditer(text):
            commands.append(("read_file", {"path": match.group(1)}))

        # Inline commands: "use the `find` command", "run `ls -la src/`"
        for match in self.INLINE_CMD.finditer(text):
            cmd = match.group(1).strip()
            if cmd:
                commands.append(self._parse_cmd(cmd))

        # Backtick commands: "`ls -la`", "`cat src/app.py`"
        for match in self.BACKTICK_CMD.finditer(text):
            cmd = match.group(1).strip()
            if cmd:
                commands.append(self._parse_cmd(cmd))

        return self._augment_with_prose(text, commands)

    def _parse_cmd(self, cmd: str) -> tuple[str, dict]:
        if cmd.startswith(("ls", "dir", "find")):
            return ("list_files", {"path": "."})
        elif cmd.startswith(("cat ", "head ", "tail ")):
            parts = cmd.split()
            return ("read_file", {"path": parts[-1]}) if len(parts) >= 2 else ("run_command", {"command": cmd})
        elif ">" in cmd and "echo" in cmd:
            parts = cmd.split(">")
            return (
                "write_file",
                {
                    "path": parts[1].strip(),
                    "content": parts[0].replace("echo", "").strip().strip("\"'"),
                },
            )
        return ("run_command", {"command": cmd, "timeout": 30})


class GemmaParser(BaseParser):
    """Parser for Gemma-family models (Gemma-4 E2B, E4B).

    Gemma models use different output formats:
    - <code>...</code> blocks instead of ```bash
    - Plain command text without code fences
    - Function call syntax: tool_name(arg="value")
    - Markdown with `inline code` for commands
    - Numbered step lists with commands embedded
    """

    CODE_TAG = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL)
    INLINE_CODE = re.compile(r"`([^`]+)`")
    STEP_LIST = re.compile(
        r'\d+\.\s+(?:Run|Execute|Use|Type|Enter|Run the command)[:\s]+[`"]?([^`"\n]+)[`"]?',
        re.IGNORECASE,
    )
    PLAIN_CMD = re.compile(
        r'(?:^|\n)\s*(?:I\'ll |Let me |Now )?(?:run|execute|use|type|check|list|read|view|show|cat|ls|find|grep)\s+(?:the command\s+)?[`"]?([^`"\n.]+)[`]?',
        re.IGNORECASE | re.MULTILINE,
    )
    FUNC_CALL = re.compile(r"(\w+)\(([^)]*)\)")

    def extract_commands(self, text: str) -> list[tuple[str, dict]]:
        text = _strip_stop_tokens(text)
        commands = []

        # <code> blocks
        for match in self.CODE_TAG.finditer(text):
            block = match.group(1).strip()
            for line in block.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    commands.append(self._parse_cmd(line))

        # Inline code (single backtick)
        for match in self.INLINE_CODE.finditer(text):
            candidate = match.group(1).strip()
            if self._looks_like_command(candidate):
                commands.append(self._parse_cmd(candidate))

        # Step lists: "1. Run: ls -la"
        for match in self.STEP_LIST.finditer(text):
            cmd = match.group(1).strip()
            if cmd:
                commands.append(self._parse_cmd(cmd))

        # Plain command references
        for match in self.PLAIN_CMD.finditer(text):
            cmd = match.group(1).strip()
            if cmd and len(cmd) < 100:  # Avoid matching long prose
                commands.append(self._parse_cmd(cmd))

        # Function call syntax: list_files(path=".")
        for match in self.FUNC_CALL.finditer(text):
            func_name = match.group(1)
            args_str = match.group(2)
            if func_name in (
                "list_files",
                "read_file",
                "write_file",
                "run_command",
                "fetch_url",
                "search_web",
                "browse_page",
                "send_request",
            ):
                args = self._parse_func_args(args_str)
                commands.append((func_name, args))

        return self._augment_with_prose(text, commands)

    def _looks_like_command(self, text: str) -> bool:
        """Check if a string looks like a shell command."""
        cmd_prefixes = (
            "ls",
            "cat",
            "find",
            "grep",
            "head",
            "tail",
            "cd",
            "pwd",
            "mkdir",
            "rm",
            "cp",
            "mv",
            "chmod",
            "chown",
            "df",
            "du",
            "ps",
            "kill",
            "top",
            "wget",
            "curl",
            "ssh",
            "scp",
            "tar",
            "python",
            "node",
            "npm",
            "git",
            "docker",
            "make",
            "gcc",
            "./",
            "../",
            "/",
            "~/",
        )
        return any(text.startswith(p) for p in cmd_prefixes) or "/" in text

    def _parse_cmd(self, cmd: str) -> tuple[str, dict]:
        cmd = cmd.strip("`\"' ")
        if cmd.startswith(("ls", "dir")):
            return ("list_files", {"path": "."})
        elif cmd.startswith(("cat ", "head ", "tail ")):
            parts = cmd.split()
            return ("read_file", {"path": parts[-1]}) if len(parts) >= 2 else ("run_command", {"command": cmd})
        elif cmd.startswith(("find ", "grep ")):
            return ("run_command", {"command": cmd, "timeout": 30})
        elif ">" in cmd and "echo" in cmd:
            parts = cmd.split(">")
            return (
                "write_file",
                {
                    "path": parts[1].strip(),
                    "content": parts[0].replace("echo", "").strip().strip("\"'"),
                },
            )
        return ("run_command", {"command": cmd, "timeout": 30})

    def _parse_func_args(self, args_str: str) -> dict:
        """Parse function call arguments like: path=".", command="ls" """
        args = {}
        for match in re.finditer(r'(\w+)\s*=\s*["\']?([^"\',\s]+)["\']?', args_str):
            args[match.group(1)] = match.group(2)
        return args


class BonsaiParser(BaseParser):
    """Parser for Bonsai 1-bit models.

    These tiny models produce less structured output. They may:
    - Write commands without any formatting
    - Repeat themselves
    - Produce shorter, less coherent responses
    """

    CMD_PATTERN = re.compile(
        r"(?:^|\n)\s*((?:cat|ls|find|grep|head|tail|cd|mkdir|rm|cp|mv|python|node|git|curl|wget)\s+\S+(?:\s+\S+)*)",
        re.MULTILINE,
    )
    FILE_REF = re.compile(r"(\w[\w/]*\.\w{1,4})")

    def extract_commands(self, text: str) -> list[tuple[str, dict]]:
        text = _strip_stop_tokens(text)
        commands = []

        # Direct command matches
        for match in self.CMD_PATTERN.finditer(text):
            cmd = match.group(1).strip()
            if cmd:
                commands.append(self._parse_cmd(cmd))

        # File references that look like they're being accessed
        for match in self.FILE_REF.finditer(text):
            path = match.group(1)
            if "/" in path or path.endswith((".py", ".js", ".txt", ".md", ".json", ".yaml", ".yml")):
                commands.append(("read_file", {"path": path}))

        return self._augment_with_prose(text, commands)

    def _parse_cmd(self, cmd: str) -> tuple[str, dict]:
        if cmd.startswith(("ls", "find")):
            return ("list_files", {"path": "."})
        elif cmd.startswith(("cat ", "head ", "tail ")):
            parts = cmd.split()
            return ("read_file", {"path": parts[-1]}) if len(parts) >= 2 else ("run_command", {"command": cmd})
        return ("run_command", {"command": cmd, "timeout": 30})


class ProseHeavyParser(QwenParser):
    """Parser for models that lean on prose more than code blocks (Nemotron, TinyLlama).

    Inherits Qwen's code-block extraction and relies on the prose augmentor
    in BaseParser to catch natural-language commands.
    """

    pass


def get_parser(model_name: str) -> BaseParser:
    """Get the appropriate parser for a model."""
    model_lower = model_name.lower()
    if "gemma" in model_lower:
        return GemmaParser()
    if "bonsai" in model_lower:
        return BonsaiParser()
    if "nemotron" in model_lower or "tinyllama" in model_lower:
        return ProseHeavyParser()
    return QwenParser()
