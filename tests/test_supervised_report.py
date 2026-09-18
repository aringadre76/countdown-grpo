from countdown_grpo.supervised_report import diagnostic_gate


def test_decreasing_loss_does_not_pass_without_legal_dev_outputs():
    metrics = [{"loss": 1, "grad_norm": 2}, {"loss": 0.3, "grad_norm": 1}]
    gate = diagnostic_gate(metrics, {"legal_expression_rate": 0.1})
    assert gate["loss_decreased"]
    assert not all(gate.values())
    metrics[1]["grad_norm"] = float("nan")
    assert not diagnostic_gate(metrics, {"legal_expression_rate": 0.6})["finite_logged_gradients"]
