import json
from pathlib import Path

from simpnmr_x_synth.app.pipelines.dataset_validation import validate_dataset_case


def test_validate_dataset_case_writes_truth_vs_fit_report(tmp_path: Path):
    case_dir = tmp_path / "cases" / "case"
    truth = case_dir / "DATA" / "CHI"
    fitted = case_dir / "SIMULATIONS" / "FITTING" / "simpnmr_x_fitted_output"
    truth.mkdir(parents=True)
    fitted.mkdir(parents=True)
    (truth / "susceptibility.csv").write_text(
        "chi_xx,chi_xy,chi_xz,chi_yy,chi_yz,chi_zz\n1,2,3,4,5,6\n",
        encoding="utf-8",
    )
    (tmp_path / "dataset.csv").write_text(
        "sample_id,linewidth_p1,linewidth_p2\ncase,7,8\n", encoding="utf-8"
    )
    (fitted / "susceptibility_tensor.csv").write_text(
        (
            "Temperature (K),chi_xx (Å^3),chi_xy (Å^3),chi_xz (Å^3),"
            "chi_yy (Å^3),chi_yz (Å^3),chi_zz (Å^3)\n"
            "302.15,1.5,2,3,4,5,6\n"
        ),
        encoding="utf-8",
    )
    (fitted / "linewidth_model_302.15_K.csv").write_text(
        "linewidth_method,p1,p2\nr6,7,8.25\n", encoding="utf-8"
    )
    (fitted / "moment_fit_diagnostics_302.15_K.csv").write_text(
        "# score = 0.125\nquantity,m1\nobserved,1\n", encoding="utf-8"
    )

    report_file = validate_dataset_case(case_dir)
    report = json.loads(report_file.read_text(encoding="utf-8"))

    assert report["status"] == "compared"
    assert report["absolute_error"]["chi_xx"] == 0.5
    assert report["absolute_error"]["linewidth_p2"] == 0.25
    assert report["moment_score"] == 0.125
