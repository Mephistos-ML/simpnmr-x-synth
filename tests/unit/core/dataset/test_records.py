import pytest

from simpnmr_x_synth.core.dataset.records import DatasetRecord, TensorTarget


def test_dataset_record_preserves_dynamic_ordered_moment_vector():
    record = DatasetRecord(
        sample_id="sample-000001-302K",
        temperature_k=302.15,
        magnetic_field_t=4.7,
        moments={"m1": 1.0, "m2": 2.0, "m3": 3.0, "m4": 4.0},
        target=TensorTarget(1.0, 0.1, 0.2, 2.0, 0.3, 3.0, 705.05, 0.25),
    )

    assert list(record.moments) == ["m1", "m2", "m3", "m4"]
    assert record.target.as_row() == {
        "chi_xx": 1.0,
        "chi_xy": 0.1,
        "chi_xz": 0.2,
        "chi_yy": 2.0,
        "chi_yz": 0.3,
        "chi_zz": 3.0,
        "linewidth_p1": 705.05,
        "linewidth_p2": 0.25,
    }


def test_dataset_record_rejects_nonconsecutive_moment_names():
    with pytest.raises(ValueError, match="ordered consecutively"):
        DatasetRecord(
            sample_id="sample-1",
            temperature_k=300.0,
            magnetic_field_t=4.7,
            moments={"m1": 1.0, "m3": 3.0},
            target=TensorTarget(1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 705.05, 0.25),
        )
