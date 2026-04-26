"""
Tests for tools/train_cloud.py — portable Colab-runnable CNN training script.

Verifies:
  - DEFAULT_MASK matches the prod _TRAINING_CONSTANT_CHANNELS set
  - save_prod_model writes the {arch, n_channels, state_dict} dict format
    that backend's CNNAgent.load_model expects
  - write_best_loss writes a single float as a string (matching prod
    cnn_agent._BEST_LOSS_PATH format)
  - write_progress_json writes {status, epoch_log, ...} matching the
    backend hot-reload contract
  - SignalCNN.forward accepts (B, 27, 60) and returns (B, 1)
  - apply_mask zeros only the channels listed in mask
"""
import os
import sys
import json
import pytest
import torch

# tools/ is sibling of backend/
_TOOLS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)


def _import_module():
    import importlib
    import train_cloud  # type: ignore
    importlib.reload(train_cloud)
    return train_cloud


class TestDefaultMaskMatchesProd:
    def test_default_mask_is_frozenset_of_prod_constant_channels(self):
        tc = _import_module()
        assert tc.DEFAULT_MASK == frozenset({10, 11, 20, 24, 25, 26})

    def test_default_mask_is_frozenset_type(self):
        tc = _import_module()
        assert isinstance(tc.DEFAULT_MASK, frozenset)


class TestSignalCNNShape:
    def test_signalcnn_forward_returns_b1_for_b_27_60_input(self):
        tc = _import_module()
        m = tc.SignalCNN(n_ch=27)
        m.eval()
        x = torch.zeros(4, 27, 60)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (4, 1)

    def test_signalcnn_arch_is_glu2(self):
        tc = _import_module()
        m = tc.SignalCNN(n_ch=27)
        assert getattr(m, "arch", None) == "glu2"


class TestApplyMask:
    def test_apply_mask_zeros_listed_channels_only(self):
        tc = _import_module()
        x = torch.ones(2, 27, 60)
        out = tc.apply_mask(x, frozenset({10, 24}))
        assert torch.all(out[:, 10, :] == 0.0)
        assert torch.all(out[:, 24, :] == 0.0)
        assert torch.all(out[:, 11, :] == 1.0)
        assert torch.all(out[:, 0, :] == 1.0)

    def test_apply_mask_empty_set_is_passthrough(self):
        tc = _import_module()
        x = torch.ones(1, 27, 60) * 0.5
        out = tc.apply_mask(x, frozenset())
        assert torch.equal(out, x)


class TestSaveProdModel:
    def test_save_prod_model_writes_arch_n_channels_state_dict(self, tmp_path):
        tc = _import_module()
        m = tc.SignalCNN(n_ch=27)
        target = tmp_path / "cnn_model.pt"
        tc.save_prod_model(m, str(target), n_channels=27)
        assert target.exists()
        blob = torch.load(str(target), map_location="cpu", weights_only=False)
        assert isinstance(blob, dict)
        assert blob["arch"] == "glu2"
        assert blob["n_channels"] == 27
        assert "state_dict" in blob
        # state_dict round-trips back into the model
        m2 = tc.SignalCNN(n_ch=27)
        m2.load_state_dict(blob["state_dict"])


class TestWriteBestLoss:
    def test_write_best_loss_writes_float_as_string(self, tmp_path):
        tc = _import_module()
        target = tmp_path / "cnn_best_loss.txt"
        tc.write_best_loss(str(target), 0.6712)
        assert target.exists()
        v = float(target.read_text().strip())
        assert abs(v - 0.6712) < 1e-9


class TestWriteProgressJson:
    def test_progress_has_status_and_epoch_log(self, tmp_path):
        tc = _import_module()
        target = tmp_path / "train_cloud_progress.json"
        epoch_log = [
            {"epoch": 1, "train_loss": 0.69, "val_loss": 0.70, "lr": 1e-3},
            {"epoch": 2, "train_loss": 0.66, "val_loss": 0.67, "lr": 1e-3},
        ]
        tc.write_progress_json(
            str(target),
            status="completed",
            epoch_log=epoch_log,
            best_val_loss=0.67,
            n_train=12345,
            n_val=2000,
        )
        d = json.loads(target.read_text())
        assert d["status"] == "completed"
        assert d["epoch_log"] == epoch_log
        assert d["best_val_loss"] == 0.67
        assert d["n_train"] == 12345
        assert d["n_val"] == 2000

    def test_progress_running_status_writes_partial_log(self, tmp_path):
        tc = _import_module()
        target = tmp_path / "p.json"
        tc.write_progress_json(str(target), status="running", epoch_log=[])
        d = json.loads(target.read_text())
        assert d["status"] == "running"
        assert d["epoch_log"] == []
