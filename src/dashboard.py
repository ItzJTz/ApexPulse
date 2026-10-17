import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

if __package__:
    from .analysis import (
        align_driver_sessions,
        align_laps,
        load_track_config,
        optimal_lap_analysis,
        sector_results,
    )
    from .telemetry_io import (
        fastest_lap_numbers,
        load_telemetry,
        load_telemetry_text,
        process_session,
    )
else:
    from analysis import (
        align_driver_sessions,
        align_laps,
        load_track_config,
        optimal_lap_analysis,
        sector_results,
    )
    from telemetry_io import (
        fastest_lap_numbers,
        load_telemetry,
        load_telemetry_text,
        process_session,
    )


@st.cache_data(show_spinner=False)
def _load_local(path, modified_time):
    del modified_time
    return load_telemetry(path)


@st.cache_data(show_spinner=False)
def _load_upload(content, filename):
    return load_telemetry_text(content, filename)


def _channel_trace(frame, lap_number, distance, channel):
    lap = frame[frame["Lap Number"] == lap_number].copy()
    lap[channel] = pd.to_numeric(lap[channel], errors="coerce")
    lap = (
        lap.groupby("Normalized_Dist", as_index=False)[channel]
        .mean()
        .sort_values("Normalized_Dist")
    )
    values = lap[channel]
    valid = values.notna()
    if not valid.any():
        return np.full(len(distance), np.nan)
    return np.interp(
        distance,
        lap.loc[valid, "Normalized_Dist"],
        values.loc[valid],
    )


def _input_figure(frame, lap_a, lap_b, alignment):
    available = [
        channel
        for channel in ("Throttle Pos", "Brake Pos", "Steering", "G Force Lat", "G Force Long")
        if channel in frame.columns
        and frame[channel].notna().any()
    ]
    if not available:
        return None
    distance = alignment["distance_m"]
    figure = make_subplots(
        rows=len(available),
        cols=1,
        shared_xaxes=True,
        subplot_titles=available,
        vertical_spacing=min(0.08, 0.3 / len(available)),
    )
    for row, channel in enumerate(available, start=1):
        figure.add_trace(
            go.Scatter(
                x=distance,
                y=_channel_trace(frame, lap_a, distance, channel),
                mode="lines",
                name=f"Lap {lap_a:g}",
                legendgroup="lap_a",
                line={"color": "#00d2ff"},
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=distance,
                y=_channel_trace(frame, lap_b, distance, channel),
                mode="lines",
                name=f"Lap {lap_b:g}",
                legendgroup="lap_b",
                line={"color": "#ff007f"},
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
        figure.update_yaxes(title_text=channel, row=row, col=1)
    figure.update_xaxes(title_text="Lap distance (m)", row=len(available), col=1)
    figure.update_layout(
        template="plotly_dark",
        height=max(480, 190 * len(available)),
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        legend={"orientation": "h", "y": 1.04, "x": 1, "xanchor": "right"},
    )
    return figure


def _gg_figure(lap):
    if "G Force Lat" not in lap or "G Force Long" not in lap:
        return None
    data = lap.copy()
    data["G Force Lat"] = pd.to_numeric(data["G Force Lat"], errors="coerce")
    data["G Force Long"] = pd.to_numeric(data["G Force Long"], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["G Force Lat", "G Force Long"])
    if data.empty:
        return None
    lateral = data["G Force Lat"].to_numpy(dtype=float)
    longitudinal = -data["G Force Long"].to_numpy(dtype=float)
    density, lateral_edges, longitudinal_edges = np.histogram2d(
        lateral,
        longitudinal,
        bins=45,
    )
    density = np.where(density.T > 0, density.T, np.nan)
    lateral_centers = (lateral_edges[:-1] + lateral_edges[1:]) / 2
    longitudinal_centers = (longitudinal_edges[:-1] + longitudinal_edges[1:]) / 2
    figure = go.Figure(
        go.Heatmap(
            x=lateral_centers,
            y=longitudinal_centers,
            z=density,
            colorscale="Inferno",
            colorbar={"title": "Samples/bin"},
            hovertemplate=(
                "Lateral: %{x:.2f} G<br>Longitudinal: %{y:.2f} G"
                "<br>Samples/bin: %{z:.0f}<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=0, line_dash="dot", line_color="#8b949e")
    figure.add_vline(x=0, line_dash="dot", line_color="#8b949e")
    figure.update_layout(
        template="plotly_dark",
        title="G-G sample density · empty bins hidden",
        xaxis_title="Lateral acceleration (G)",
        yaxis_title="Acceleration / braking (G)",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return figure


def _track_figure(lap):
    speed = lap["Ground Speed"].to_numpy(dtype=float)
    latitude = (
        pd.to_numeric(lap["GPS Latitude"], errors="coerce")
        if "GPS Latitude" in lap
        else None
    )
    longitude = (
        pd.to_numeric(lap["GPS Longitude"], errors="coerce")
        if "GPS Longitude" in lap
        else None
    )
    if latitude is not None and longitude is not None:
        gps = lap.assign(_latitude=latitude, _longitude=longitude).dropna(
            subset=["_latitude", "_longitude"]
        )
        if (
            len(gps) >= 3
            and gps["_latitude"].between(-90, 90).all()
            and gps["_longitude"].between(-180, 180).all()
            and gps[["_latitude", "_longitude"]].drop_duplicates().shape[0] >= 3
            and gps["_latitude"].std() > 1e-6
            and gps["_longitude"].std() > 1e-6
        ):
            figure = px.scatter(
                gps,
                x="_longitude",
                y="_latitude",
                color="Ground Speed",
                color_continuous_scale="Magma",
                hover_data=["Normalized_Dist", "Ground Speed"],
                title="Logged GPS position (coordinates not independently validated)",
            )
            figure.update_traces(mode="lines+markers", marker={"size": 4})
            figure.update_layout(
                template="plotly_dark",
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                yaxis={"scaleanchor": "x", "scaleratio": 1},
            )
            return figure, "Logged GPS coordinates"

    if "G Force Lat" not in lap or lap["G Force Lat"].isna().all():
        return None, "No usable GPS position or lateral-acceleration channel"
    distance = lap["Normalized_Dist"].to_numpy(dtype=float)
    if len(distance) < 3 or distance[-1] <= 0:
        return None, "Not enough spatial samples to reconstruct a map"
    ds = np.diff(distance, prepend=distance[0])
    speed_ms = np.maximum(speed / 3.6, 2.0)
    lateral_g = pd.to_numeric(lap["G Force Lat"], errors="coerce").fillna(0).to_numpy()
    heading = np.cumsum((lateral_g * 9.81 / speed_ms**2) * ds)
    x_raw = np.cumsum(np.cos(heading) * ds)
    y_raw = np.cumsum(np.sin(heading) * ds)
    progress = distance / distance[-1]
    x = x_raw - progress * (x_raw[-1] - x_raw[0])
    y = y_raw - progress * (y_raw[-1] - y_raw[0])
    figure = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker={
                "size": 5,
                "color": speed,
                "colorscale": "Magma",
                "colorbar": {"title": "km/h"},
            },
            customdata=np.column_stack((distance, speed)),
            hovertemplate="Distance: %{customdata[0]:.0f} m<br>Speed: %{customdata[1]:.1f} km/h<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_dark",
        title="Approximate path reconstructed from lateral G (not surveyed geometry)",
        xaxis_title="Relative X (m)",
        yaxis_title="Relative Y (m)",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
    )
    return figure, "Approximate reconstruction from lateral acceleration"


def main():
    st.set_page_config(page_title="ApexPulse Telemetry", page_icon="🏁", layout="wide")
    st.title("ApexPulse")
    st.caption("Motorsport telemetry analysis · compare laps, inspect channels, and review data quality")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
        [data-testid="stMetric"] {background: #161b22; border: 1px solid #30363d;
            padding: 0.85rem 1rem; border-radius: 0.6rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    data_directory = Path(__file__).resolve().parent.parent / "data"
    local_files = sorted(data_directory.glob("*.csv"))
    preferred = next(
        (index for index, path in enumerate(local_files) if path.name.startswith("raw_")),
        0,
    )
    with st.sidebar:
        st.header("Session")
        upload = st.file_uploader("Upload a telemetry CSV", type=("csv",))
        local_path = None
        if local_files:
            selected_name = st.selectbox(
                "Or choose a local CSV",
                [path.name for path in local_files],
                index=preferred,
                disabled=upload is not None,
            )
            local_path = data_directory / selected_name
        minimum_fraction = st.slider(
            "Minimum lap coverage",
            min_value=0.5,
            max_value=1.0,
            value=0.9,
            step=0.01,
            help="Laps shorter than this fraction of the configured or observed track length are excluded.",
        )

    if upload is None and local_path is None:
        st.info("Upload a telemetry CSV or place one in the repository's data folder.")
        return
    try:
        with st.spinner("Loading and validating telemetry..."):
            frame = (
                _load_upload(upload.getvalue(), upload.name)
                if upload is not None
                else _load_local(str(local_path), local_path.stat().st_mtime_ns)
            )
            clean_frame, quality = process_session(frame, minimum_fraction)
    except (OSError, TypeError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not analyze this telemetry file: {exc}")
        return

    lap_options = fastest_lap_numbers(
        clean_frame,
        count=max(2, clean_frame["Lap Number"].nunique()),
    )
    with st.sidebar:
        if len(lap_options) >= 2:
            default_a = lap_options[0]
            default_b = lap_options[1]
            lap_a = st.selectbox(
                "Reference lap",
                lap_options,
                index=lap_options.index(default_a),
                format_func=lambda value: f"Lap {value:g}",
            )
            lap_b_options = [lap for lap in lap_options if lap != lap_a]
            lap_b = st.selectbox(
                "Compared lap",
                lap_b_options,
                index=lap_b_options.index(default_b) if default_b in lap_b_options else 0,
                format_func=lambda value: f"Lap {value:g}",
            )
        else:
            lap_a = lap_options[0] if lap_options else None
            lap_b = None

    track_label = quality.get("track") or "Track not identified"
    st.subheader(f"{track_label} · {Path(quality['source_file']).name}")
    if not quality.get("track"):
        st.warning("No matching track configuration was detected. Lap comparisons work; track sectors are unavailable.")
    if lap_b is None:
        st.warning("At least two complete laps are needed to compare laps.")

    metrics = st.columns(4)
    retained_rows = quality["rows"]["after_lap_filter"]
    retained_display = (
        f"{retained_rows / 1000:.1f}k"
        if retained_rows >= 10000
        else f"{retained_rows:,}"
    )
    track_length = quality["reference_track_length_m"]
    track_length_display = (
        f"{track_length / 1000:.1f} km"
        if track_length >= 1000
        else f"{track_length:,.0f} m"
    )
    metrics[0].metric("Valid laps", quality["laps"]["total"] - len(quality["laps"]["excluded_incomplete"]))
    metrics[1].metric("Samples kept", retained_display)
    metrics[2].metric("Invalid rows", f"{quality['rows']['invalid_required_data']:,}")
    metrics[3].metric("Circuit length", track_length_display)

    alignment = align_laps(clean_frame, lap_a, lap_b) if lap_b is not None else None
    tabs = st.tabs(
        (
            "Overview",
            "Driver inputs",
            "G-G",
            "Track map",
            "Sectors",
            "Data quality",
            "Driver comparison",
            "Optimal lap",
        )
    )

    with tabs[0]:
        if alignment is None:
            st.info("Choose a session with at least two complete laps to see the comparison.")
        else:
            recorded_gap = alignment["lap_a_seconds"] - alignment["lap_b_seconds"]
            spatial_delta = float(alignment["delta_b_ahead_s"][-1])
            overview = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                subplot_titles=("Ground speed", "Time delta · positive means compared lap is ahead"),
                vertical_spacing=0.12,
            )
            overview.add_trace(
                go.Scatter(x=alignment["distance_m"], y=alignment["speed_a_kmh"], name=f"Lap {lap_a:g}", line={"color": "#00d2ff"}),
                row=1,
                col=1,
            )
            overview.add_trace(
                go.Scatter(x=alignment["distance_m"], y=alignment["speed_b_kmh"], name=f"Lap {lap_b:g}", line={"color": "#ff007f"}),
                row=1,
                col=1,
            )
            overview.add_trace(
                go.Scatter(x=alignment["distance_m"], y=alignment["delta_b_ahead_s"], name="Compared lap ahead", line={"color": "#00ffcc"}, fill="tozeroy"),
                row=2,
                col=1,
            )
            overview.add_hline(y=0, line_dash="dot", line_color="#8b949e", row=2, col=1)
            overview.update_xaxes(title_text="Lap distance (m)", row=2, col=1)
            overview.update_yaxes(title_text="km/h", row=1, col=1)
            overview.update_yaxes(title_text="seconds", row=2, col=1)
            overview.update_layout(template="plotly_dark", height=650, margin={"l": 20, "r": 20, "t": 60, "b": 20})
            st.plotly_chart(overview, width="stretch")
            time_metrics = st.columns(4)
            time_metrics[0].metric(f"Lap {lap_a:g}", f"{alignment['lap_a_seconds']:.3f} s")
            time_metrics[1].metric(f"Lap {lap_b:g}", f"{alignment['lap_b_seconds']:.3f} s")
            time_metrics[2].metric(
                "Recorded lap-time gap",
                f"{recorded_gap:+.3f} s",
                help="Positive means the reference lap's recorded time is faster.",
            )
            time_metrics[3].metric(
                "Distance-integrated delta",
                f"{spatial_delta:+.3f} s",
                help="Positive means the compared lap is faster.",
            )
            st.caption(
                "The lap-time gap uses recorded elapsed time when available; the plotted "
                "distance delta is independently estimated from speed and distance."
            )

    with tabs[1]:
        if alignment is not None:
            inputs = _input_figure(clean_frame, lap_a, lap_b, alignment)
            if inputs is None:
                st.info("No throttle, brake, steering, or acceleration channels are available.")
            else:
                st.plotly_chart(inputs, width="stretch")
        else:
            st.info("Select at least two complete laps to compare driver inputs.")

    with tabs[2]:
        selected_lap = clean_frame[clean_frame["Lap Number"] == lap_a]
        gg = _gg_figure(selected_lap) if lap_a is not None else None
        if gg is None:
            st.info("This session does not contain usable lateral and longitudinal G channels.")
        else:
            st.plotly_chart(gg, width="stretch")
            st.markdown(
                "Each colored cell represents telemetry samples in a small acceleration range. "
                "Brighter cells mean more samples were recorded there; empty ranges are left blank. "
                "Lateral G is signed left/right. Longitudinal G is sign-inverted here, so positive "
                "values mean acceleration and negative values mean braking."
            )
            lat = pd.to_numeric(selected_lap["G Force Lat"], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            long = -pd.to_numeric(
                selected_lap["G Force Long"], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).dropna()
            force_metrics = st.columns(3)
            force_metrics[0].metric("Peak lateral |G|", f"{lat.abs().max():.2f} G")
            force_metrics[1].metric("99th percentile |lateral G|", f"{lat.abs().quantile(0.99):.2f} G")
            force_metrics[2].metric("Peak displayed longitudinal G", f"{long.abs().max():.2f} G")
            st.caption("These are measured channel summaries, not a tire-grip limit or validated utilization score.")

    with tabs[3]:
        map_lap = clean_frame[clean_frame["Lap Number"] == lap_a] if lap_a is not None else clean_frame.iloc[0:0]
        track_figure, method = _track_figure(map_lap)
        if track_figure is None:
            st.info(method)
        else:
            st.caption(method)
            st.plotly_chart(track_figure, width="stretch")

    with tabs[4]:
        config = load_track_config(clean_frame)
        if alignment is None:
            st.info("Select two complete laps to compare sectors.")
        elif config is None:
            st.info("No matching track configuration is available for sector analysis.")
        else:
            sectors = sector_results(clean_frame, lap_a, lap_b, config)
            available_sectors = [sector for sector in sectors if sector["available"]]
            if available_sectors:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Sector": sector["sector"],
                                f"Lap {lap_a:g} V-min (km/h)": sector["lap_a_v_min_kmh"],
                                f"Lap {lap_b:g} V-min (km/h)": sector["lap_b_v_min_kmh"],
                                "V-min delta (km/h)": sector["v_min_delta_kmh"],
                                "Lap A time (s)": sector["lap_a_seconds"],
                                "Lap B time (s)": sector["lap_b_seconds"],
                                "Delta, B ahead (s)": sector["delta_b_ahead_seconds"],
                            }
                            for sector in available_sectors
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.caption("Sector timings are estimates integrated over the configured distance gates.")
            else:
                st.info("The selected laps do not cover the configured sector gates.")

    with tabs[5]:
        rows = st.columns(4)
        rows[0].metric("Input rows", f"{quality['rows']['source']:,}")
        rows[1].metric("Invalid required rows", f"{quality['rows']['invalid_required_data']:,}")
        rows[2].metric("Repeated-distance samples retained", f"{quality['rows']['repeated_distance_samples']:,}")
        rows[3].metric("Partial laps excluded", str(len(quality["laps"]["excluded_incomplete"])))
        st.subheader("Lap coverage")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Lap": float(lap),
                        "Distance (m)": distance,
                        "Included": float(lap) in quality["laps"]["complete"],
                    }
                    for lap, distance in quality["laps"]["distance_m"].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        missing_channels = quality["channels"]["optional_missing_percent"]
        if missing_channels:
            st.subheader("Optional channel completeness")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Channel": channel, "Missing (%)": missing}
                        for channel, missing in missing_channels.items()
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        report = quality if alignment is None else {
            "schema_version": 1,
            "data_quality": quality,
            "analysis": {
                "reference_lap": float(lap_a),
                "compared_lap": float(lap_b),
                "reference_duration_seconds": alignment["lap_a_seconds"],
                "compared_duration_seconds": alignment["lap_b_seconds"],
                "delta_compared_lap_ahead_seconds": float(alignment["delta_b_ahead_s"][-1]),
                "delta_method": "trapezoidal distance/speed integration",
                "sector_results": (
                    sector_results(clean_frame, lap_a, lap_b, load_track_config(clean_frame))
                    if load_track_config(clean_frame)
                    else []
                ),
                "optimal_lap": optimal_lap_analysis(clean_frame),
            },
        }
        report_bytes = json.dumps(report, indent=2).encode("utf-8")
        st.download_button(
            "Download session report (JSON)",
            data=report_bytes,
            file_name=f"{Path(quality['source_file']).stem}_session_report.json",
            mime="application/json",
        )
        st.download_button(
            "Download cleaned telemetry (CSV)",
            data=clean_frame.to_csv(index=False).encode("utf-8"),
            file_name=f"{Path(quality['source_file']).stem}_cleaned.csv",
            mime="text/csv",
        )

    with tabs[6]:
        st.markdown(
            "Compare the fastest complete lap in this session with the fastest complete lap "
            "from a second telemetry file. The files should be from the same circuit and use "
            "the same distance units."
        )
        comparison_upload = st.file_uploader(
            "Upload the second driver's telemetry CSV",
            type=("csv",),
            key="driver_comparison_upload",
        )
        if comparison_upload is None:
            st.info("Upload a second driver's CSV to show the cross-session comparison.")
        else:
            try:
                comparison_frame = _load_upload(
                    comparison_upload.getvalue(),
                    comparison_upload.name,
                )
                comparison_frame, comparison_quality = process_session(
                    comparison_frame,
                    minimum_fraction,
                )
                primary_track = quality.get("track")
                comparison_track = comparison_quality.get("track")
                if not primary_track or not comparison_track:
                    st.warning(
                        "A track could not be confirmed from both filenames or Venue metadata. "
                        "Verify these sessions are from the same circuit and use the same distance units."
                    )
                    confirmed = st.checkbox(
                        "I confirm both sessions are from the same circuit and use the same distance units.",
                        key="driver_comparison_track_confirmation",
                    )
                else:
                    confirmed = True
                    st.caption(f"Both sessions detected as {primary_track}.")

                if confirmed:
                    comparison = align_driver_sessions(clean_frame, comparison_frame)
                    figure = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=(
                            "Ground speed",
                            "Time delta · positive means Driver B is ahead",
                        ),
                        vertical_spacing=0.12,
                    )
                    figure.add_trace(
                        go.Scatter(
                            x=comparison["distance_m"],
                            y=comparison["speed_a_kmh"],
                            name=f"Session A · lap {comparison['lap_a']:g}",
                            line={"color": "#00d2ff"},
                        ),
                        row=1,
                        col=1,
                    )
                    figure.add_trace(
                        go.Scatter(
                            x=comparison["distance_m"],
                            y=comparison["speed_b_kmh"],
                            name=f"Session B · lap {comparison['lap_b']:g}",
                            line={"color": "#ff007f"},
                        ),
                        row=1,
                        col=1,
                    )
                    figure.add_trace(
                        go.Scatter(
                            x=comparison["distance_m"],
                            y=comparison["delta_b_ahead_s"],
                            name="Session B ahead",
                            line={"color": "#00ffcc"},
                            fill="tozeroy",
                        ),
                        row=2,
                        col=1,
                    )
                    figure.add_hline(y=0, line_dash="dot", line_color="#8b949e", row=2, col=1)
                    figure.update_xaxes(title_text="Lap distance (m)", row=2, col=1)
                    figure.update_yaxes(title_text="km/h", row=1, col=1)
                    figure.update_yaxes(title_text="seconds", row=2, col=1)
                    figure.update_layout(
                        template="plotly_dark",
                        height=650,
                        margin={"l": 20, "r": 20, "t": 60, "b": 20},
                    )
                    st.plotly_chart(figure, width="stretch")
                    comparison_metrics = st.columns(3)
                    comparison_metrics[0].metric(
                        f"Session A lap {comparison['lap_a']:g}",
                        f"{comparison['lap_a_seconds']:.3f} s",
                    )
                    comparison_metrics[1].metric(
                        f"Session B lap {comparison['lap_b']:g}",
                        f"{comparison['lap_b_seconds']:.3f} s",
                    )
                    comparison_metrics[2].metric(
                        "Distance-integrated delta",
                        f"{comparison['delta_b_ahead_s'][-1]:+.3f} s",
                        help="Positive means Session B is estimated to be faster over the shared distance.",
                    )
                    st.caption(
                        "Lap durations use recorded time when available; the plotted delta is a "
                        "separate estimate integrated from speed over the shared distance."
                    )
            except (OSError, TypeError, ValueError, pd.errors.ParserError) as exc:
                st.error(f"Could not compare these telemetry sessions: {exc}")

    with tabs[7]:
        micro_sector_meters = st.slider(
            "Micro-sector length (m)",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Smaller sectors give a finer theoretical combination but are more sensitive to sampling noise.",
        )
        try:
            optimal = optimal_lap_analysis(clean_frame, micro_sector_meters)
            optimal_metrics = st.columns(3)
            optimal_metrics[0].metric(
                f"Best complete lap · {optimal['actual_best_lap']:g}",
                f"{optimal['actual_best_seconds']:.3f} s",
            )
            optimal_metrics[1].metric(
                "Theoretical best",
                f"{optimal['theoretical_best_seconds']:.3f} s",
            )
            optimal_metrics[2].metric(
                "Estimated potential gain",
                f"{optimal['potential_gain_seconds']:.3f} s",
            )
            sector_frame = pd.DataFrame(optimal["sectors"])
            sector_frame["Distance"] = sector_frame.apply(
                lambda sector: f"{sector['start_m']:.0f}-{sector['end_m']:.0f} m",
                axis=1,
            )
            figure = go.Figure(
                go.Bar(
                    x=sector_frame["Distance"],
                    y=sector_frame["best_time_seconds"],
                    marker={
                        "color": sector_frame["best_lap"],
                        "colorscale": "Turbo",
                        "showscale": True,
                        "colorbar": {"title": "Best lap"},
                    },
                    customdata=sector_frame["best_lap"],
                    hovertemplate=(
                        "Distance: %{x}<br>Best sector time: %{y:.3f} s"
                        "<br>From lap %{customdata:g}<extra></extra>"
                    ),
                )
            )
            figure.update_layout(
                template="plotly_dark",
                title="Fastest observed time in each distance micro-sector",
                xaxis_title="Distance micro-sector",
                yaxis_title="Estimated time (s)",
            )
            st.plotly_chart(figure, width="stretch")
            st.caption(
                f"Estimates use distance/speed integration over {optimal['common_distance_m']:.0f} m. "
                "The theoretical lap combines the fastest sector from different laps and is a "
                "benchmark, not a lap achieved in one run."
            )
        except ValueError as exc:
            st.info(str(exc))


if __name__ == "__main__":
    main()
