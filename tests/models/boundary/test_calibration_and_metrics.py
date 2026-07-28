"""Calibration fitting and length/absent diagnostic tests."""

import torch
import torch.nn.functional as F

from gliner2.configuration import validate_boundary_head
from gliner2.joint_ie.calibration import (
    expected_calibration_error,
    fit_binary_temperature,
)
from gliner2.training.trainer import ExtractorTrainer


def test_fitted_temperature_reduces_dev_nll_and_ece():
    torch.manual_seed(7)
    true_logits = torch.linspace(-2.5, 2.5, 600)
    targets = torch.bernoulli(torch.sigmoid(true_logits))
    overconfident_logits = true_logits * 3.0
    before_nll = F.binary_cross_entropy_with_logits(
        overconfident_logits, targets
    )
    before_ece = expected_calibration_error(
        torch.sigmoid(overconfident_logits), targets
    )
    temperature = fit_binary_temperature(overconfident_logits, targets)
    after_nll = F.binary_cross_entropy_with_logits(
        overconfident_logits / temperature, targets
    )
    after_ece = expected_calibration_error(
        torch.sigmoid(overconfident_logits / temperature), targets
    )
    assert temperature > 1.0
    assert after_nll < before_nll
    assert after_ece < before_ece


def test_temperature_and_negative_sampling_config_validation():
    values = validate_boundary_head(
        {
            "pair_temperature": 1.7,
            "relation_temperature": 1.2,
            "record_temperature": 0.9,
            "classification_temperature": 1.1,
            "negative_query_ratio": 1.0,
            "max_negative_queries_per_batch": 8,
        }
    )
    assert values["pair_temperature"] == 1.7
    assert values["negative_query_ratio"] == 1.0


def test_public_metric_ratios_include_length_and_absent_queries():
    metrics = ExtractorTrainer._proposal_metric_ratios(
        {
            "length_1_hit": torch.tensor(3),
            "length_1_total": torch.tensor(4),
            "length_9_plus_hit": torch.tensor(1),
            "length_9_plus_total": torch.tensor(2),
            "absent_query_false_positive": torch.tensor(2),
            "absent_query_total": torch.tensor(10),
        },
        "eval",
    )
    assert metrics["eval_recall_length_1"] == 0.75
    assert metrics["eval_recall_length_9_plus"] == 0.5
    assert metrics["eval_absent_query_false_positive_rate"] == 0.2
