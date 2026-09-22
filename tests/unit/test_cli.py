from simpnmr_x_synth.cli.main import build_parser


def test_cli_exposes_only_dataset_generation_and_validation_commands():
    parser = build_parser()

    generate = parser.parse_args(["dataset", "generate", "config.yml"])
    validate = parser.parse_args(["dataset", "validate", "case-dir"])

    assert generate.command == "dataset"
    assert generate.dataset_command == "generate"
    assert validate.command == "dataset"
    assert validate.dataset_command == "validate"
