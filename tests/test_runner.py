"""Tests for pipeline runner and checkpoint."""


from deconfounding_interp.runner import Checkpoint


def test_checkpoint_roundtrip(tmp_path):
    cp = Checkpoint(tmp_path / "cp.json")
    assert not cp.is_completed("job_1")
    cp.mark_completed("job_1", {"status": "ok"})
    assert cp.is_completed("job_1")
    assert not cp.is_completed("job_2")


def test_checkpoint_persistence(tmp_path):
    path = tmp_path / "cp.json"
    cp1 = Checkpoint(path)
    cp1.mark_completed("job_1", {"status": "ok"})

    cp2 = Checkpoint(path)
    assert cp2.is_completed("job_1")


def test_checkpoint_stores_timestamp(tmp_path):
    cp = Checkpoint(tmp_path / "cp.json")
    cp.mark_completed("job_1", {"status": "ok"})
    assert "timestamp" in cp.data["completed"]["job_1"]
