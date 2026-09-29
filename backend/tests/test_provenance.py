"""A stored score must be attributable to the artifacts and config that produced it.

Blocker 1 of the 2026-09-28 controlling document: `cnn_scans` and `trades` carry no model or
config identity, so no measurement in this repository can be attributed to a version. The peer
session made the same point independently — "agent tag is not model provenance" — and it is why
a 1,582-trade PnL figure silently mixed model eras.

This is the pure half: turn "what was loaded" into one stable string. No database, no schema,
no clock. Persisting it is a separate change, because a migration is a different kind of risk
than a hash function.

The properties that matter are all about FAILING LOUD rather than producing a plausible-looking
digest, because a fingerprint that is wrong is worse than none: it invites attribution that
cannot be justified.
"""

from __future__ import annotations

import os
import sys

import pytest

_BACKEND = os.path.join(os.path.dirname(__file__), "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from services.provenance import artifact_fingerprint  # noqa: E402


def _write(tmp_path, name, content=b"model-bytes"):
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


def test_identical_inputs_give_an_identical_digest(tmp_path):
    a = _write(tmp_path, "m.json")
    one = artifact_fingerprint([a], {"feature_set": "v3"})
    two = artifact_fingerprint([a], {"feature_set": "v3"})
    assert one["digest"] == two["digest"]
    assert one["digest"].startswith("sha256:")


def test_changing_one_byte_of_an_artifact_changes_the_digest(tmp_path):
    a = _write(tmp_path, "m.json", b"model-bytes")
    before = artifact_fingerprint([a], {"feature_set": "v3"})["digest"]
    (tmp_path / "m.json").write_bytes(b"model-byteS")
    after = artifact_fingerprint([a], {"feature_set": "v3"})["digest"]
    assert before != after


def test_changing_a_config_value_changes_the_digest(tmp_path):
    a = _write(tmp_path, "m.json")
    v3 = artifact_fingerprint([a], {"feature_set": "v3"})["digest"]
    v4 = artifact_fingerprint([a], {"feature_set": "v4"})["digest"]
    assert v3 != v4


def test_the_digest_does_not_depend_on_the_order_paths_are_supplied(tmp_path):
    a = _write(tmp_path, "a.json", b"aaa")
    b = _write(tmp_path, "b.json", b"bbb")
    assert artifact_fingerprint([a, b], {})["digest"] == artifact_fingerprint([b, a], {})["digest"]


def test_a_missing_artifact_raises_rather_than_fingerprinting_absence(tmp_path):
    """If we cannot see what was loaded we must not mint an identity for it. Returning a
    digest that silently encodes "file absent" is the failure mode that lets an
    unattributable run look attributable."""
    with pytest.raises(FileNotFoundError):
        artifact_fingerprint([str(tmp_path / "nope.json")], {})


def test_config_values_of_different_types_do_not_collide(tmp_path):
    """`True == 1` in Python, and a naive str() or dict-key comparison lets a bool alias an
    int. The peer session found exactly that aliasing in another artifact path
    (`schema_version=True` comparing equal to 1), so it is pinned here."""
    a = _write(tmp_path, "m.json")
    digests = {
        artifact_fingerprint([a], {"k": v})["digest"] for v in (1, True, "1", 1.0, None, "True")
    }
    assert len(digests) == 6, "distinct config values collapsed to the same digest"


def test_the_detail_names_every_input_so_a_mismatch_is_diagnosable(tmp_path):
    """A bare digest tells you two runs differ but never why. Provenance has to survive
    contact with an investigation."""
    a = _write(tmp_path, "m.json", b"xyz")
    got = artifact_fingerprint([a], {"feature_set": "v3", "threshold": 0.6})
    assert os.path.basename(a) in str(got["artifacts"])
    assert got["config"] == {"feature_set": "v3", "threshold": 0.6}
    per_file = got["artifacts"][0]
    assert per_file["bytes"] == 3
    assert per_file["sha256"] and per_file["sha256"] != got["digest"]


def test_an_empty_input_set_is_refused(tmp_path):
    """Fingerprinting nothing would yield a constant that looks like an identity."""
    with pytest.raises(ValueError):
        artifact_fingerprint([], {})


def test_a_directory_is_not_mistaken_for_an_artifact(tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(ValueError):
        artifact_fingerprint([str(d)], {})
