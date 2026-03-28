import geochemad.benchmark as benchmark_module
from geochemad.config import BenchmarkConfig, PreprocessConfig
from geochemad.models.statistical import ZScore


def test_run_benchmark_smoke_with_synthetic_data(monkeypatch, synthetic_data_dir, tmp_path):
    monkeypatch.setattr(
        benchmark_module,
        "get_all_models",
        lambda config, device: {"ZS": ZScore()},
    )

    config = BenchmarkConfig(
        data_dir=str(synthetic_data_dir),
        output_dir=str(tmp_path / "results"),
        datasets=["rock2"],
        num_runs=1,
        random_seed=11,
        device="cpu",
        preprocess=PreprocessConfig(transform="raw"),
    )

    results = benchmark_module.run_benchmark(
        config=config,
        model_names=["ZS"],
        verbose=False,
    )

    assert "rock2" in results.index
    assert "ZS" in results.columns
    assert (tmp_path / "results" / "benchmark_results.csv").exists()
    assert (tmp_path / "results" / "detailed_results.json").exists()
