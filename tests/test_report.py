import json

from countdown_grpo.report import write_svg


def test_svg_is_generated_from_observed_proportions(tmp_path):
    output = tmp_path / "figure.svg"
    write_svg(output, [("base test", 0.25), ("missing", None)])
    text = output.read_text()
    assert "base test" in text
    assert "25.00%" in text
    assert "not measured" in text
    assert json.loads("{\"valid\": true}")["valid"]
