import pandas as pd

from final_model.eval.benchmark_compare import compare_references


def test_compare_preserves_metrics_and_does_not_create_scalar_score(tmp_path):
    human = tmp_path / "human.csv"
    model = tmp_path / "model.csv"
    output = tmp_path / "comparison.csv"
    pd.DataFrame([{
        "source": "human", "Time": 200, "Horizon": None,
        "aoi_contract": None, "metric": "latency", "value": 80,
        "numerator": None, "denominator": 10,
    }]).to_csv(human, index=False)
    pd.DataFrame([{
        "source": "model", "Time": 200, "Horizon": 2,
        "aoi_contract": None, "metric": "latency", "value": 100,
        "numerator": None, "denominator": 12,
    }]).to_csv(model, index=False)
    result = compare_references(
        human_reference=human, model_reference=model, output=output
    )
    assert result.loc[0, "difference_model_minus_human"] == 20
    assert result.loc[0, "comparison_status"] == "available"
    assert "score" not in result.columns
