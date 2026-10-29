from pathlib import Path


from scripts.alod.alod_ext_prevalence_association import (
    DEFAULT_CONFIG,
    build_config,
    capacity_from_union,
    dataset_spec,
    load_study,
)


def test_capacity_rule_reproduces_run009_practical_point() -> None:
    study = load_study(DEFAULT_CONFIG)
    spec = dataset_spec(study, "sciart")
    assert capacity_from_union(study, spec, 3_503_048) == 3_700_000


def test_capacity_rule_caps_at_available_hierarchy_rows() -> None:
    study = load_study(DEFAULT_CONFIG)
    spec = dataset_spec(study, "campus")
    assert capacity_from_union(study, spec, spec["hierarchy_rows"]) == spec["hierarchy_rows"]


def test_dataset_config_preserves_selector_and_mode_boundaries() -> None:
    study = load_study(DEFAULT_CONFIG)
    spec = dataset_spec(study, "sciart")
    nocache = build_config(spec, "nocache")
    previous = build_config(spec, "previous")
    bounded = build_config(spec, "bounded", 1_000_000)

    for config in (nocache, previous, bounded):
        assert config["alod"]["spt_root_volume"] == spec["spt_root_volume"]
        assert config["alod"]["min_spt_size"] == 100
        assert config["alod"]["target_granularity"] == spec["target_granularity"]
        assert config["trajectory_id"] == spec["trajectory_id"]
        assert Path(config["camera_path"]) == Path(spec["camera_path"])

    assert nocache["alod"]["materialization_filter"] == "none"
    assert nocache["alod_host"]["mode"] == "host_nocache"
    assert previous["alod"]["materialization_filter"] == "conservative_pinhole_gpu"
    assert previous["alod_host"]["mode"] == "host_previous_reuse"
    assert bounded["alod_host"]["mode"] == "host_bounded_cache"
    assert bounded["alod_host"]["cache_capacity_rows"] == 1_000_000
