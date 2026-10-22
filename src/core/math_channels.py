import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.special import gamma

from src.config import config_value
from src.pipeline.telemetry_io import (
    fastest_lap_numbers,
    load_telemetry,
    process_session,
)
from src.visualization.style import apply_style


def tire_metrics(frame):
    """Summarize optional tire signals and measured per-lap G-G envelopes."""
    exponent = float(config_value("tire_analysis", "friction_circle_exponent"))
    if not np.isfinite(exponent) or exponent <= 0:
        raise ValueError("tire_analysis.friction_circle_exponent must be positive and finite.")
    areas = {}
    work = {}
    thermal = None
    unavailable = {}

    required_temperature = {"Tire Temp Inner", "Tire Temp Outer"}
    if required_temperature.issubset(frame.columns):
        inner = pd.to_numeric(frame["Tire Temp Inner"], errors="coerce")
        outer = pd.to_numeric(frame["Tire Temp Outer"], errors="coerce")
        gradient = inner - outer
        if gradient.notna().any():
            thermal = {
                "gradient_c": gradient.to_numpy(dtype=float),
                "lateral_g": (
                    pd.to_numeric(frame["G Force Lat"], errors="coerce").to_numpy(
                        dtype=float
                    )
                    if "G Force Lat" in frame.columns
                    else np.full(len(frame), np.nan)
                ),
            }
    else:
        unavailable["thermal_gradient"] = (
            "Inner and outer tire-temperature channels are required."
        )
    if thermal is None and required_temperature.issubset(frame.columns):
        unavailable["thermal_gradient"] = (
            "Tire-temperature channels contain no usable paired samples."
        )

    if {"G Force Lat", "G Force Long"}.issubset(frame.columns):
        for lap_number, lap in frame.groupby("Lap Number", sort=True):
            lateral = pd.to_numeric(lap["G Force Lat"], errors="coerce").to_numpy(
                dtype=float
            )
            longitudinal = -pd.to_numeric(
                lap["G Force Long"], errors="coerce"
            ).to_numpy(dtype=float)
            finite = np.isfinite(lateral) & np.isfinite(longitudinal)
            lateral = lateral[finite]
            longitudinal = longitudinal[finite]
            if len(lateral) < 3:
                continue
            lateral_radius = float(np.max(np.abs(lateral)))
            accel_radius = float(np.quantile(longitudinal, 0.99))
            brake_radius = float(abs(np.min(longitudinal)))
            if min(lateral_radius, accel_radius, brake_radius) <= 0:
                continue
            gamma_factor = gamma(1 + 1 / exponent) ** 2 / gamma(1 + 2 / exponent)
            areas[float(lap_number)] = float(
                2
                * lateral_radius
                * (accel_radius + brake_radius)
                * gamma_factor
            )
    else:
        unavailable["friction_circle_area"] = (
            "Lateral and longitudinal G channels are required."
        )

    required_work = {"Steering", "Ground Speed", "G Force Lat", "Lap Distance"}
    if required_work.issubset(frame.columns):
        for lap_number, lap in frame.groupby("Lap Number", sort=True):
            values = pd.DataFrame(
                {
                    "distance": pd.to_numeric(
                        lap["Normalized_Dist"]
                        if "Normalized_Dist" in lap.columns
                        else lap["Lap Distance"],
                        errors="coerce",
                    ),
                    "steering": pd.to_numeric(lap["Steering"], errors="coerce"),
                    "speed": pd.to_numeric(lap["Ground Speed"], errors="coerce") / 3.6,
                    "lateral_g": pd.to_numeric(lap["G Force Lat"], errors="coerce"),
                },
                index=lap.index,
            ).replace([np.inf, -np.inf], np.nan).dropna()
            values = values.sort_values("distance").drop_duplicates("distance")
            if len(values) < 2:
                continue
            if "Time" in lap.columns:
                elapsed = pd.to_numeric(
                    lap.loc[values.index, "Time"], errors="coerce"
                ).to_numpy(dtype=float, copy=True)
                elapsed -= elapsed[0]
                if (
                    not np.isfinite(elapsed).all()
                    or np.any(np.diff(elapsed) <= 0)
                ):
                    elapsed = None
            else:
                elapsed = None
            if elapsed is None:
                distance = values["distance"].to_numpy(dtype=float)
                speed = np.maximum(values["speed"].to_numpy(dtype=float), 1.0)
                elapsed = np.concatenate(
                    ([0.0], np.cumsum(np.diff(distance) / speed[1:]))
                )
            integrand = (
                values["steering"].abs().to_numpy(dtype=float)
                * values["speed"].abs().to_numpy(dtype=float)
                * values["lateral_g"].abs().to_numpy(dtype=float)
            )
            work[float(lap_number)] = float(trapezoid(integrand, x=elapsed))
    else:
        unavailable["tire_work_integral"] = (
            "Steering, speed, lateral G, and lap distance channels are required."
        )

    area_comparison = None
    if areas:
        available_laps = sorted(areas)
        configured_baseline = config_value("tire_analysis", "baseline_lap")
        configured_compare = config_value("tire_analysis", "compare_lap")
        baseline_lap = (
            available_laps[0]
            if configured_baseline is None
            else float(configured_baseline)
        )
        compare_lap = (
            available_laps[-1]
            if configured_compare is None
            else float(configured_compare)
        )
        missing_laps = [
            lap
            for lap in (baseline_lap, compare_lap)
            if lap not in areas
        ]
        if missing_laps:
            unavailable["friction_circle_area_comparison"] = (
                "Configured G-G comparison lap(s) are unavailable: "
                + ", ".join(f"{lap:g}" for lap in missing_laps)
                + f". Available laps: {', '.join(f'{lap:g}' for lap in available_laps)}."
            )
        elif areas[baseline_lap] <= 0:
            unavailable["friction_circle_area_comparison"] = (
                f"Baseline lap {baseline_lap:g} has no positive G-G envelope area."
            )
        else:
            area_comparison = {
                "baseline_lap": baseline_lap,
                "compare_lap": compare_lap,
                "change_percent": float(
                    (areas[compare_lap] - areas[baseline_lap])
                    / areas[baseline_lap]
                    * 100
                ),
            }
    else:
        unavailable["friction_circle_area_comparison"] = (
            "No valid per-lap G-G envelope areas are available."
        )
    return {
        "friction_circle_area_g2_by_lap": areas,
        "friction_circle_area_comparison": area_comparison,
        "tire_work_integral_by_lap": work,
        "thermal_gradient": thermal,
        "unavailable": unavailable,
    }


def generate_math_channels(csv_path="data/clean_data.csv"):
    df, _ = process_session(load_telemetry(csv_path))
    speed_col = "Ground Speed"
    tire_summary = tire_metrics(df)
    selected_laps = fastest_lap_numbers(df, count=1)
    if not selected_laps:
        raise ValueError("No complete laps are available for derived-channel analysis.")
    df = df[df["Lap Number"] == selected_laps[0]].copy()
    df = df.sort_values(by="Normalized_Dist").reset_index(drop=True)

    # 2. Dynamic Brake Bias (%)
    brake_col = "Brake Pos" if "Brake Pos" in df.columns else next(
        (
            channel
            for channel in df.columns
            if "brake" in channel.lower() and "pressure" not in channel.lower()
        ),
        None,
    )
    if brake_col:
        df["Math_BrakeBias"] = np.where(
            df[brake_col] > float(config_value("vehicle", "brake_active_threshold")),
            float(config_value("vehicle", "default_brake_bias")),
            0.0,
        )
    else:
        df["Math_BrakeBias"] = 0.0

    # 3. Aero Downforce Estimation (N)
    v_ms = df[speed_col] / 3.6
    c_aero = float(config_value("vehicle", "aero_downforce_mult"))
    df["Math_AeroDownforce_N"] = c_aero * (v_ms ** 2)
    if {"Tire Temp Inner", "Tire Temp Outer"}.issubset(df.columns):
        df["Math_TireTempGradient_C"] = (
            pd.to_numeric(df["Tire Temp Inner"], errors="coerce")
            - pd.to_numeric(df["Tire Temp Outer"], errors="coerce")
        )
    selected_lap = float(selected_laps[0])
    df["Math_FrictionCircleArea_G2"] = tire_summary[
        "friction_circle_area_g2_by_lap"
    ].get(selected_lap, np.nan)
    df["Math_TireWorkIntegral"] = tire_summary[
        "tire_work_integral_by_lap"
    ].get(selected_lap, np.nan)
    area_comparison = tire_summary["friction_circle_area_comparison"]
    df["Math_FrictionCircleAreaChange_Pct"] = (
        area_comparison["change_percent"]
        if area_comparison is not None
        else np.nan
    )

    # Save cleaned CSV with math channels
    df.to_csv("data/data_with_math_channels.csv", index=False)

    # 4. Plot Dark-Theme Dashboard
    DARK_BG = "#0e1117"
    apply_style()
    has_thermal = (
        "Math_TireTempGradient_C" in df.columns
        and "G Force Lat" in df.columns
    )
    axes_count = 3 if has_thermal else 2
    fig, axes = plt.subplots(
        axes_count,
        1,
        figsize=(12, 4 * axes_count),
        sharex=True,
    )
    ax1, ax2 = axes[:2]
    fig.patch.set_facecolor(DARK_BG)

    for ax in axes:
        ax.grid(True)

    # Speed vs Downforce
    ax1.plot(df["Normalized_Dist"], df[speed_col], color="#00ff66", label="Filtered Ground Speed (km/h)", linewidth=1.5)
    ax1.set_ylabel("Speed (km/h)", color="#c9d1d9", fontweight="bold")
    ax1.set_title("ApexPulse | Vehicle Dynamics Math Channels (Derived Aero Downforce)", color="#ffffff", fontweight="bold", pad=15)
    ax1.legend(loc="upper right", facecolor=DARK_BG, edgecolor="#30363d", labelcolor="#ffffff")

    ax2.plot(df["Normalized_Dist"], df["Math_AeroDownforce_N"], color="#ff0055", label="Calculated Aero Downforce (N)", linewidth=1.8)
    ax2.set_xlabel("Track Distance (m)", color="#c9d1d9", fontweight="bold")
    ax2.set_ylabel("Downforce (N)", color="#c9d1d9", fontweight="bold")
    ax2.legend(loc="upper right", facecolor=DARK_BG, edgecolor="#30363d", labelcolor="#ffffff")
    if has_thermal:
        thermal_axis = axes[2]
        thermal_axis.scatter(
            pd.to_numeric(df["G Force Lat"], errors="coerce"),
            df["Math_TireTempGradient_C"],
            color="#00d2ff",
            s=8,
            alpha=0.55,
        )
        thermal_axis.set_xlabel("Lateral G")
        thermal_axis.set_ylabel("Inner - outer tire temp (°C)")
        thermal_axis.set_title("Tire thermal gradient vs lateral acceleration")

    plt.tight_layout()
    plt.savefig("assets/math_channels.png", dpi=300, facecolor=DARK_BG)
    print("Exported smooth, noise-filtered dashboard to assets/math_channels.png")
    plt.show()

if __name__ == "__main__":
    generate_math_channels()