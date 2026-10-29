"""Streamlit dashboard for gsplat/viser viewer metrics CSV logs.

Run:
    streamlit run tools/viewer_metrics_dashboard.py -- --logs-dir logs/viewer_metrics

Expected log layout:
    logs/viewer_metrics/
      naive_x1.csv
      naive_x4.csv
      culled_x9.csv

Install:
    python -m pip install -r tools/requirements-viewer-dashboard.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import altair as alt
import pandas as pd
import streamlit as st


RENDER_COLUMNS = {
    "render_fps",
    "render_frame_time_ms",
    "render_width",
    "render_height",
    "active_blocks",
    "active_gaussians",
    "visible_gaussians",
    "total_gaussians",
    "resident_blocks",
    "resident_gaussians",
    "cache_hits",
    "cache_misses",
    "cache_hit_rate",
    "newly_uploaded_blocks",
    "newly_uploaded_gaussians",
    "evicted_blocks",
    "evicted_gaussians",
    "gpu_resident_bytes",
    "gpu_resident_mb",
    "cpu_resident_bytes",
    "cpu_resident_mb",
    "gpu_upload_ms",
    "block_cull_ms",
    "block_concat_ms",
    "peak_vram",
    "peak_vram_mb",
    "torch_allocated_mb",
    "torch_reserved_mb",
    "torch_frame_peak_allocated_mb",
    "torch_frame_peak_reserved_mb",
    "gpu_budget_exceeded",
    "active_set_exceeds_budget",
}

SYSTEM_COLUMNS = {
    "process_cpu_percent",
    "process_ram_gb",
    "system_ram_used_gb",
    "system_ram_percent",
    "gpu_util_percent",
    "vram_used_gb",
    "vram_total_gb",
    "vram_allocated_gb",
    "vram_reserved_gb",
    "vram_max_allocated_gb",
}

IMPORTANT_COLUMNS = [
    "elapsed_sec",
    "render_frame_index",
    "render_fps",
    "render_frame_time_ms",
    "vram_used_gb",
    "torch_frame_peak_allocated_mb",
    "system_ram_used_gb",
    "gpu_util_percent",
    "process_cpu_percent",
]

LABELS = {
    "elapsed_sec": "Elapsed Time (s)",
    "time_sec": "Time (s)",
    "render_fps": "FPS",
    "render_frame_time_ms": "Frame Time (ms)",
    "vram_used_gb": "VRAM Used (GB)",
    "vram_total_gb": "VRAM Total (GB)",
    "system_ram_used_gb": "System RAM Used (GB)",
    "system_ram_percent": "System RAM (%)",
    "process_ram_gb": "Process RAM (GB)",
    "process_cpu_percent": "Process CPU (%)",
    "gpu_util_percent": "GPU Utilization (%)",
    "active_gaussians": "Active Gaussians",
    "visible_gaussians": "Rendered / Visible Gaussians",
    "total_gaussians": "Total Gaussians",
    "active_blocks": "Active Blocks",
    "resident_blocks": "Resident Blocks",
    "resident_gaussians": "Resident Gaussians",
    "cache_hit_rate": "Cache Hit Rate",
    "newly_uploaded_blocks": "Newly Uploaded Blocks",
    "newly_uploaded_gaussians": "Newly Uploaded Gaussians",
    "evicted_blocks": "Evicted Blocks",
    "evicted_gaussians": "Evicted Gaussians",
    "gpu_upload_ms": "GPU Upload Time (ms)",
    "block_cull_ms": "Block Cull Time (ms)",
    "block_concat_ms": "Block Concat Time (ms)",
    "gpu_resident_mb": "Persistent GPU Block Cache (MiB)",
    "cpu_resident_mb": "CPU-Resident Blocks (MiB)",
    "peak_vram_mb": "Cumulative Peak Torch Allocation (MiB)",
    "torch_allocated_mb": "Live Torch Allocation (MiB)",
    "torch_reserved_mb": "Torch Reserved / Cache (MiB)",
    "torch_frame_peak_allocated_mb": "Per-Frame Peak Torch Allocation (MiB)",
    "torch_frame_peak_reserved_mb": "Per-Frame Peak Torch Reserved (MiB)",
    "gpu_budget_exceeded": "GPU Cache Budget Exceeded",
    "active_set_exceeds_budget": "Active Set Exceeds Budget",
    "peak_gpu_resident_mb": "Peak Persistent GPU Cache (MiB)",
    "peak_torch_allocated_mb": "Peak Live Torch Allocation (MiB)",
    "avg_torch_allocated_mb": "Average Live Torch Allocation (MiB)",
    "peak_torch_reserved_mb": "Peak Torch Reserved / Cache (MiB)",
    "avg_torch_reserved_mb": "Average Torch Reserved / Cache (MiB)",
    "peak_torch_frame_peak_allocated_mb": "Peak Per-Frame Torch Allocation (MiB)",
    "avg_torch_frame_peak_allocated_mb": "Average Per-Frame Torch Allocation (MiB)",
    "peak_torch_frame_peak_reserved_mb": "Peak Per-Frame Torch Reserved (MiB)",
    "avg_cache_hit_rate": "Average Cache Hit Rate",
    "peak_cuda_vram_mb": "Cumulative Peak Torch Allocation (MiB)",
    "total_uploaded_blocks": "Total Uploaded Blocks",
    "total_evicted_blocks": "Total Evicted Blocks",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--logs-dir", type=str, default="logs/viewer_metrics")
    return parser.parse_known_args()[0]


def discover_csvs(logs_dir: Path) -> list[Path]:
    if not logs_dir.exists():
        return []
    return sorted(path for path in logs_dir.rglob("*.csv") if path.is_file())


def make_run_name(path: Path, logs_dir: Path) -> str:
    try:
        rel = path.relative_to(logs_dir)
    except ValueError:
        rel = path.name
    return str(rel).removesuffix(".csv")


@st.cache_data(show_spinner=False)
def load_csv(path: str, run_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["run"] = run_name
    df["source_file"] = path
    for col in df.columns:
        if col in RENDER_COLUMNS or col in SYSTEM_COLUMNS or col in {
            "timestamp",
            "elapsed_sec",
            "pid",
            "render_frame_index",
        }:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "elapsed_sec" in df.columns:
        df["time_sec"] = df["elapsed_sec"] - df["elapsed_sec"].min()
    elif "timestamp" in df.columns:
        df["time_sec"] = df["timestamp"] - df["timestamp"].min()
    else:
        df["time_sec"] = range(len(df))

    if "render_frame_index" in df.columns:
        frame_idx = pd.to_numeric(df["render_frame_index"], errors="coerce")
        df["new_render_frame"] = (
            frame_idx.notna() & frame_idx.gt(0) & frame_idx.ne(frame_idx.shift())
        )
    else:
        df["new_render_frame"] = pd.NA
    return df


def available_columns(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [col for col in candidates if col in df.columns and df[col].notna().any()]


def percentile_1_low(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return float("nan")
    return float(values.quantile(0.01))


def first_valid_value(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return float("nan")
    return float(values.iloc[-1])


def summarize_run(sample_df: pd.DataFrame, frame_df: pd.DataFrame, run: str) -> dict:
    duration = float(sample_df["time_sec"].max() - sample_df["time_sec"].min())
    summary = {
        "run": run,
        "duration_s": duration,
        "rendered_frames": int(frame_df["render_frame_index"].nunique())
        if "render_frame_index" in frame_df
        else len(frame_df),
        "avg_fps": frame_df["render_fps"].mean()
        if "render_fps" in frame_df
        else float("nan"),
        "median_fps": frame_df["render_fps"].median()
        if "render_fps" in frame_df
        else float("nan"),
        "1pct_low_fps": percentile_1_low(frame_df["render_fps"])
        if "render_fps" in frame_df
        else float("nan"),
        "avg_frame_time_ms": frame_df["render_frame_time_ms"].mean()
        if "render_frame_time_ms" in frame_df
        else float("nan"),
        "median_frame_time_ms": frame_df["render_frame_time_ms"].median()
        if "render_frame_time_ms" in frame_df
        else float("nan"),
        "peak_frame_time_ms": frame_df["render_frame_time_ms"].max()
        if "render_frame_time_ms" in frame_df
        else float("nan"),
        "peak_vram_gb": sample_df["vram_used_gb"].max()
        if "vram_used_gb" in sample_df
        else float("nan"),
        "avg_vram_gb": sample_df["vram_used_gb"].mean()
        if "vram_used_gb" in sample_df
        else float("nan"),
        "peak_ram_gb": sample_df["system_ram_used_gb"].max()
        if "system_ram_used_gb" in sample_df
        else float("nan"),
        "avg_gpu_util_percent": sample_df["gpu_util_percent"].mean()
        if "gpu_util_percent" in sample_df
        else float("nan"),
        "total_gaussians": first_valid_value(frame_df["total_gaussians"])
        if "total_gaussians" in frame_df
        else float("nan"),
        "avg_active_gaussians": frame_df["active_gaussians"].mean()
        if "active_gaussians" in frame_df
        else float("nan"),
        "peak_active_gaussians": frame_df["active_gaussians"].max()
        if "active_gaussians" in frame_df
        else float("nan"),
        "avg_rendered_gaussians": frame_df["visible_gaussians"].mean()
        if "visible_gaussians" in frame_df
        else float("nan"),
        "peak_rendered_gaussians": frame_df["visible_gaussians"].max()
        if "visible_gaussians" in frame_df
        else float("nan"),
        "avg_active_blocks": frame_df["active_blocks"].mean()
        if "active_blocks" in frame_df
        else float("nan"),
        "peak_active_blocks": frame_df["active_blocks"].max()
        if "active_blocks" in frame_df
        else float("nan"),
        "avg_resident_blocks": frame_df["resident_blocks"].mean()
        if "resident_blocks" in frame_df
        else float("nan"),
        "peak_resident_blocks": frame_df["resident_blocks"].max()
        if "resident_blocks" in frame_df
        else float("nan"),
        "avg_resident_gaussians": frame_df["resident_gaussians"].mean()
        if "resident_gaussians" in frame_df
        else float("nan"),
        "peak_resident_gaussians": frame_df["resident_gaussians"].max()
        if "resident_gaussians" in frame_df
        else float("nan"),
        "avg_cache_hit_rate": frame_df["cache_hit_rate"].mean()
        if "cache_hit_rate" in frame_df
        else float("nan"),
        "total_uploaded_blocks": frame_df["newly_uploaded_blocks"].sum()
        if "newly_uploaded_blocks" in frame_df
        else float("nan"),
        "total_evicted_blocks": frame_df["evicted_blocks"].sum()
        if "evicted_blocks" in frame_df
        else float("nan"),
        "peak_gpu_resident_mb": frame_df["gpu_resident_mb"].max()
        if "gpu_resident_mb" in frame_df
        else float("nan"),
        "peak_cuda_vram_mb": frame_df["peak_vram_mb"].max()
        if "peak_vram_mb" in frame_df
        else float("nan"),
        "peak_torch_allocated_mb": frame_df["torch_allocated_mb"].max()
        if "torch_allocated_mb" in frame_df
        else float("nan"),
        "avg_torch_allocated_mb": frame_df["torch_allocated_mb"].mean()
        if "torch_allocated_mb" in frame_df
        else float("nan"),
        "peak_torch_reserved_mb": frame_df["torch_reserved_mb"].max()
        if "torch_reserved_mb" in frame_df
        else float("nan"),
        "avg_torch_reserved_mb": frame_df["torch_reserved_mb"].mean()
        if "torch_reserved_mb" in frame_df
        else float("nan"),
        "peak_torch_frame_peak_allocated_mb": frame_df[
            "torch_frame_peak_allocated_mb"
        ].max()
        if "torch_frame_peak_allocated_mb" in frame_df
        else float("nan"),
        "avg_torch_frame_peak_allocated_mb": frame_df[
            "torch_frame_peak_allocated_mb"
        ].mean()
        if "torch_frame_peak_allocated_mb" in frame_df
        else float("nan"),
        "peak_torch_frame_peak_reserved_mb": frame_df[
            "torch_frame_peak_reserved_mb"
        ].max()
        if "torch_frame_peak_reserved_mb" in frame_df
        else float("nan"),
        "budget_exceeded_frames": frame_df["gpu_budget_exceeded"].fillna(0).astype(bool).sum()
        if "gpu_budget_exceeded" in frame_df
        else float("nan"),
        "active_set_exceeded_frames": frame_df["active_set_exceeds_budget"].fillna(0).astype(bool).sum()
        if "active_set_exceeds_budget" in frame_df
        else float("nan"),
    }
    return summary


def long_chart(
    df: pd.DataFrame,
    y_cols: list[str],
    title: str,
    y_title: str,
    mark: str = "line",
) -> None:
    cols = ["run", "time_sec", *y_cols]
    chart_df = df[cols].melt(
        id_vars=["run", "time_sec"], var_name="metric", value_name="value"
    )
    chart_df = chart_df.dropna(subset=["value"])
    if chart_df.empty:
        st.info(f"No data available for {title}.")
        return

    chart_id = "".join(char if char.isalnum() else "_" for char in title.lower())
    run_selection = alt.selection_point(
        fields=["run"],
        bind="legend",
        name=f"{chart_id}_run",
        toggle=True,
    )

    base = alt.Chart(chart_df).encode(
        x=alt.X("time_sec:Q", title="Time (s)"),
        y=alt.Y("value:Q", title=y_title),
        color=alt.Color("run:N", title="Run"),
        tooltip=[
            alt.Tooltip("run:N", title="Run"),
            alt.Tooltip("metric:N", title="Metric"),
            alt.Tooltip("time_sec:Q", title="Time (s)", format=".2f"),
            alt.Tooltip("value:Q", title=y_title, format=",.2f"),
        ],
    )
    if len(y_cols) > 1:
        metric_selection = alt.selection_point(
            fields=["metric"],
            bind="legend",
            name=f"{chart_id}_metric",
            toggle=True,
        )
        active_selection = run_selection & metric_selection
        base = (
            base.encode(strokeDash=alt.StrokeDash("metric:N", title="Metric"))
            .add_params(run_selection, metric_selection)
            .encode(
                opacity=alt.condition(
                    active_selection, alt.value(1.0), alt.value(0.15)
                ),
                strokeWidth=alt.condition(
                    active_selection, alt.value(2.8), alt.value(1.1)
                ),
            )
        )
    else:
        base = base.add_params(run_selection).encode(
            opacity=alt.condition(run_selection, alt.value(1.0), alt.value(0.15)),
            strokeWidth=alt.condition(run_selection, alt.value(2.8), alt.value(1.1)),
        )

    chart = base.mark_line() if mark == "line" else base.mark_point(filled=True)
    st.altair_chart(chart.properties(title=title, height=320), use_container_width=True)


def bar_chart(summary: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
    if y_col not in summary or summary[y_col].dropna().empty:
        st.info(f"No data available for {title}.")
        return
    chart_id = "".join(char if char.isalnum() else "_" for char in title.lower())
    run_selection = alt.selection_point(
        fields=[x_col],
        bind="legend",
        name=f"{chart_id}_run",
        toggle=True,
    )
    chart = (
        alt.Chart(summary.dropna(subset=[y_col]))
        .mark_bar()
        .encode(
            x=alt.X(f"{x_col}:N", sort="-y", title="Run"),
            y=alt.Y(f"{y_col}:Q", title=LABELS.get(y_col, y_col)),
            color=alt.Color(f"{x_col}:N", title="Run"),
            opacity=alt.condition(run_selection, alt.value(1.0), alt.value(0.2)),
            tooltip=[
                alt.Tooltip(f"{x_col}:N", title="Run"),
                alt.Tooltip(f"{y_col}:Q", title=LABELS.get(y_col, y_col), format=",.2f"),
            ],
        )
        .add_params(run_selection)
        .properties(title=title, height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def scatter_chart(
    summary: pd.DataFrame, x_col: str, y_col: str, title: str, x_title: str, y_title: str
) -> None:
    if summary[[x_col, y_col]].dropna().empty:
        st.info(f"No data available for {title}.")
        return
    chart_id = "".join(char if char.isalnum() else "_" for char in title.lower())
    run_selection = alt.selection_point(
        fields=["run"],
        bind="legend",
        name=f"{chart_id}_run",
        toggle=True,
    )
    chart = (
        alt.Chart(summary.dropna(subset=[x_col, y_col]))
        .mark_circle(size=90)
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_title),
            y=alt.Y(f"{y_col}:Q", title=y_title),
            color=alt.Color("run:N", title="Run"),
            opacity=alt.condition(run_selection, alt.value(1.0), alt.value(0.2)),
            tooltip=[
                alt.Tooltip("run:N", title="Run"),
                alt.Tooltip(f"{x_col}:Q", title=x_title, format=",.2f"),
                alt.Tooltip(f"{y_col}:Q", title=y_title, format=",.2f"),
            ],
        )
        .add_params(run_selection)
        .properties(title=title, height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Viewer Metrics Dashboard", layout="wide")
    st.title("Viewer Metrics Dashboard")
    st.caption(
        "Compare gsplat/viser render performance and system resource usage across logged viewer runs."
    )
    st.caption(
        "Tip: click Run or Metric legend entries to highlight them; click again to clear. Selecting both highlights their intersection while keeping context visible."
    )

    with st.expander("Run Instructions", expanded=False):
        st.markdown(
            """
            Install dependencies:
            ```bash
            python -m pip install -r tools/requirements-viewer-dashboard.txt
            ```

            Run the dashboard:
            ```bash
            streamlit run tools/viewer_metrics_dashboard.py -- --logs-dir logs/viewer_metrics
            ```

            Expected layout: put one CSV per viewer run under `logs/viewer_metrics/`.
            Subdirectories are scanned recursively, and each file stem becomes the run label.
            """
        )

    logs_dir = Path(args.logs_dir).expanduser()
    with st.sidebar:
        st.header("Inputs")
        logs_dir_text = st.text_input("Logs directory", value=str(logs_dir))
        logs_dir = Path(logs_dir_text).expanduser()
        csv_paths = discover_csvs(logs_dir)
        if not csv_paths:
            st.warning(f"No CSV files found under {logs_dir}.")
            st.stop()

        options = {make_run_name(path, logs_dir): path for path in csv_paths}
        selected_names = st.multiselect(
            "Runs",
            options=list(options.keys()),
            default=list(options.keys())[: min(4, len(options))],
        )
        if not selected_names:
            st.warning("Select at least one CSV file.")
            st.stop()

        ignore_warmup_s = st.number_input(
            "Ignore warmup (s)", min_value=0.0, value=0.0, step=1.0
        )
        render_only = st.checkbox(
            "Render charts use only newly rendered frames",
            value=True,
            help=(
                "Uses render_frame_index changes to remove idle samples where the "
                "viewer repeats the previous FPS and frame time."
            ),
        )
        show_only_new_rows = st.checkbox(
            "Filtered table contains only newly rendered frames",
            value=False,
            help="System metrics are sampled values, so this is mainly useful for frame-level exports.",
        )

    frames = [load_csv(str(options[name]), name) for name in selected_names]
    raw_df = pd.concat(frames, ignore_index=True)
    missing = [col for col in IMPORTANT_COLUMNS if col not in raw_df.columns]
    if missing:
        st.warning(
            "Missing important columns, so related summaries or charts will be skipped: "
            + ", ".join(missing)
        )

    if "render_frame_index" not in raw_df.columns:
        st.info(
            "No render_frame_index column found. Render FPS and frame-time values are treated as sampled values, not per-frame measurements."
        )

    max_time = float(raw_df["time_sec"].max()) if raw_df["time_sec"].notna().any() else 0.0
    with st.sidebar:
        if max_time > 0:
            time_range = st.slider(
                "Time range (s)",
                min_value=0.0,
                max_value=max_time,
                value=(min(ignore_warmup_s, max_time), max_time),
                step=max(max_time / 500.0, 0.1),
            )
        else:
            time_range = (0.0, 0.0)

    filtered_df = raw_df[
        (raw_df["time_sec"] >= max(ignore_warmup_s, time_range[0]))
        & (raw_df["time_sec"] <= time_range[1])
    ].copy()

    if show_only_new_rows and "render_frame_index" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["new_render_frame"].fillna(False)]

    summary_rows = []
    render_frames_by_run = []
    for run, sample_df in filtered_df.groupby("run", sort=False):
        if "render_frame_index" in sample_df.columns and render_only:
            frame_df = sample_df[sample_df["new_render_frame"].fillna(False)].copy()
        else:
            frame_df = sample_df.copy()
        if frame_df.empty:
            frame_df = sample_df.iloc[0:0].copy()
        render_frames_by_run.append(frame_df)
        summary_rows.append(summarize_run(sample_df, frame_df, run))

    frame_df = pd.concat(render_frames_by_run, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    st.subheader("Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download filtered rows CSV",
            data=dataframe_to_csv_bytes(filtered_df),
            file_name="viewer_metrics_filtered.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Download summary CSV",
            data=dataframe_to_csv_bytes(summary_df),
            file_name="viewer_metrics_summary.csv",
            mime="text/csv",
        )

    st.subheader("Render Metrics")
    render_source = frame_df if render_only else filtered_df
    if render_only and "render_frame_index" in raw_df.columns:
        st.caption("Render plots use rows where `render_frame_index` advanced.")
    else:
        st.caption("Render plots use sampled rows; idle repeated values may be included.")

    cols = st.columns(2)
    with cols[0]:
        if "render_fps" in render_source:
            long_chart(render_source, ["render_fps"], "FPS Over Time", "FPS")
    with cols[1]:
        if "render_frame_time_ms" in render_source:
            long_chart(
                render_source,
                ["render_frame_time_ms"],
                "Frame Time Over Time",
                "Frame Time (ms)",
            )

    gaussian_cols = available_columns(
        render_source, ["total_gaussians", "active_gaussians", "visible_gaussians"]
    )
    if gaussian_cols:
        long_chart(
            render_source,
            gaussian_cols,
            "Gaussian Counts Over Time",
            "Gaussian Count",
        )
    if "active_blocks" in render_source and render_source["active_blocks"].notna().any():
        long_chart(render_source, ["active_blocks"], "Active Blocks Over Time", "Blocks")

    st.subheader("Block Cache Metrics")
    cache_count_cols = available_columns(
        render_source,
        [
            "active_blocks",
            "resident_blocks",
            "newly_uploaded_blocks",
            "evicted_blocks",
        ],
    )
    if cache_count_cols:
        long_chart(
            render_source,
            cache_count_cols,
            "Block Cache Activity Over Time",
            "Blocks",
        )

    cache_gaussian_cols = available_columns(
        render_source,
        [
            "active_gaussians",
            "resident_gaussians",
            "newly_uploaded_gaussians",
            "evicted_gaussians",
        ],
    )
    if cache_gaussian_cols:
        long_chart(
            render_source,
            cache_gaussian_cols,
            "Block Cache Gaussian Counts Over Time",
            "Gaussians",
        )

    cache_memory_cols = available_columns(
        render_source,
        ["gpu_resident_mb", "cpu_resident_mb"],
    )
    if cache_memory_cols:
        long_chart(
            render_source,
            cache_memory_cols,
            "Residency Memory Over Time",
            "MiB",
        )

    torch_memory_cols = available_columns(
        render_source,
        [
            "torch_allocated_mb",
            "torch_reserved_mb",
            "torch_frame_peak_allocated_mb",
            "torch_frame_peak_reserved_mb",
            "peak_vram_mb",
        ],
    )
    if torch_memory_cols:
        long_chart(
            render_source,
            torch_memory_cols,
            "Torch CUDA Allocator Memory Over Time",
            "MiB",
        )

    cols = st.columns(2)
    with cols[0]:
        cache_rate_cols = available_columns(render_source, ["cache_hit_rate"])
        if cache_rate_cols:
            long_chart(render_source, cache_rate_cols, "Cache Hit Rate Over Time", "Hit Rate")
        timing_cols = available_columns(
            render_source,
            ["gpu_upload_ms", "block_cull_ms", "block_concat_ms"],
        )
        if timing_cols:
            long_chart(render_source, timing_cols, "Block Pipeline Timings", "Milliseconds")
    with cols[1]:
        budget_cols = available_columns(
            render_source,
            ["gpu_budget_exceeded", "active_set_exceeds_budget"],
        )
        if budget_cols:
            long_chart(
                render_source,
                budget_cols,
                "Budget Exceeded Flags Over Time",
                "Flag",
                mark="point",
            )

    st.subheader("System Metrics")
    cols = st.columns(2)
    with cols[0]:
        if "vram_used_gb" in filtered_df:
            long_chart(filtered_df, ["vram_used_gb"], "VRAM Over Time", "VRAM (GB)")
        if "gpu_util_percent" in filtered_df:
            long_chart(
                filtered_df,
                ["gpu_util_percent"],
                "GPU Utilization Over Time",
                "GPU Utilization (%)",
            )
    with cols[1]:
        ram_cols = available_columns(
            filtered_df, ["system_ram_used_gb", "process_ram_gb"]
        )
        if ram_cols:
            long_chart(filtered_df, ram_cols, "RAM Over Time", "RAM (GB)")
        if "process_cpu_percent" in filtered_df:
            long_chart(
                filtered_df,
                ["process_cpu_percent"],
                "CPU Usage Over Time",
                "CPU (%)",
            )

    st.subheader("Run Comparisons")
    cols = st.columns(2)
    with cols[0]:
        bar_chart(summary_df, "run", "avg_fps", "Average FPS Per Run")
        bar_chart(summary_df, "run", "peak_vram_gb", "Peak VRAM Per Run")
        bar_chart(summary_df, "run", "peak_gpu_resident_mb", "Peak Persistent GPU Cache Per Run")
        bar_chart(
            summary_df,
            "run",
            "peak_torch_frame_peak_allocated_mb",
            "Peak Per-Frame Torch Allocation Per Run",
        )
    with cols[1]:
        bar_chart(summary_df, "run", "median_frame_time_ms", "Median Frame Time Per Run")
        bar_chart(summary_df, "run", "avg_vram_gb", "Average VRAM Per Run")
        bar_chart(summary_df, "run", "avg_cache_hit_rate", "Average Cache Hit Rate Per Run")
        bar_chart(summary_df, "run", "peak_torch_reserved_mb", "Peak Torch Reserved Per Run")

    cols = st.columns(2)
    with cols[0]:
        scatter_chart(
            summary_df,
            "total_gaussians",
            "avg_fps",
            "Total Gaussian Count vs Average FPS",
            "Total Gaussians",
            "Average FPS",
        )
    with cols[1]:
        scatter_chart(
            summary_df,
            "avg_active_gaussians",
            "median_frame_time_ms",
            "Active Gaussian Count vs Median Frame Time",
            "Average Active Gaussians",
            "Median Frame Time (ms)",
        )

    st.subheader("Filtered Data")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
