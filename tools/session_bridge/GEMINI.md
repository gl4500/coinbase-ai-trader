# GEMINI.md

Gemini CLI loads this file and ignores `AGENTS.md` unless `context.fileName` says
otherwise. The import below pulls in the shared directory instructions (which point to the CrewAI reference in `docs/agents/`) so every coding
assistant works from the same instructions. Keep shared conventions in `AGENTS.md`;
add Gemini-specific notes under the import.

@./AGENTS.md
