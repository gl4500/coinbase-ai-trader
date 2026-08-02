"""Regression test: main.py does not import or instantiate TechAgent (#311-refactor-c).

Locks in the TechAgent retirement. If anyone re-adds the import or the
AppState.tech_agent field without a proper revert, this test fails at
the pre-commit hook.
"""

import os
import sys

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


class TestNoTechAgentInMain:
    def test_main_module_does_not_reference_tech_agent_cb(self):
        """No import of agents.tech_agent_cb in main.py."""
        main_path = os.path.join(BACKEND, "main.py")
        src = open(main_path, encoding="utf-8").read()
        assert "tech_agent_cb" not in src, (
            "main.py references agents.tech_agent_cb — TechAgent was "
            "retired in #311-refactor-c. Re-adding requires reverting "
            "that commit, not a fresh import."
        )

    def test_app_state_has_no_tech_agent_field(self):
        """AppState dataclass does not declare a tech_agent field."""
        main_path = os.path.join(BACKEND, "main.py")
        src = open(main_path, encoding="utf-8").read()
        # Look specifically for the field declaration, not arbitrary strings
        # like comments. Both `tech_agent:` and `tech_agent =` would be flags.
        for needle in ("tech_agent:", "tech_agent =", "TechAgentCB"):
            assert needle not in src, (
                f"main.py contains '{needle}' — TechAgent was retired in #311-refactor-c."
            )
