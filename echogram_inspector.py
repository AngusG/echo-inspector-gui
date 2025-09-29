"""
Interactive Echogram Inspector using echofilter data loaders.
This application allows you to browse through echogram files and 
compare ground truth turbulence lines with generated predictions.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import pandas as pd  # Using pandas for easier CSV handling
import echofilter.raw

# Import EK500 colormap, handling case where it's already registered
# from echopype_cm import cmap_d

# --- Configuration ---
# Constants for file matching
RAW_FILE_SUFFIX = "_Sv_raw.csv"
TRUTH_TOP_FILE_SUFFIX = "_surface.evl"
TRUTH_BOT_FILE_SUFFIX = "_bottom.evl"
GEN_TOP_FILE_PREFIX = ".turbulence-"
GEN_BOT_FILE_PREFIX = ".bottom-"
MODEL_STR = "trn2025-09-04_16.27.59_bench_torch2.4_bc3784f_lr0.01_bs22_ep75_onecycle_firstpass_RTX_4090_py3.12_torch2.8.0+cu128_run1-ep75"

# Data reduction settings to avoid message size limits
MAX_PINGS = 1000  # Maximum number of pings to display
MAX_DEPTH_SAMPLES = 500  # Maximum number of depth samples
PING_DOWNSAMPLE_FACTOR = 2  # Take every Nth ping
DEPTH_DOWNSAMPLE_FACTOR = 2  # Take every Nth depth sample
MIN_DEPTH_METERS = 0.0  # Minimum depth to display (meters)
MAX_DEPTH_METERS = 5.0  # Maximum depth to display (meters)
SIGNAL_CLIP_MIN = -80  # Minimum signal value to display
SIGNAL_CLIP_MAX = -20  # Maximum signal value to display
USE_MANUAL_MAX_DEPTH = False  # Whether to force manual max depth from sidebar

# sample path
# /mnt/scratch/echohub/s3/echo-jh/ErieEastBasin2024/EB_S15_G761/

# --- CORE APPLICATION LOGIC ---


def reduce_data_size(
    signals, timestamps, depths, d_top_true, d_top_gen, d_bottom_true, d_bottom_gen, min_sv, max_sv
):
    """
    Reduce data size to avoid Streamlit message size limits.
    """
    # 1. Depth selection: keep full available depth, we'll control display range at plot time
    depth_mask = np.ones_like(depths, dtype=bool)
    signals_filtered = signals[:, depth_mask]
    depths_filtered = depths[depth_mask]

    # 2. Downsample depth samples if needed
    if len(depths_filtered) > MAX_DEPTH_SAMPLES:
        depth_indices = np.linspace(
            0, len(depths_filtered) - 1, MAX_DEPTH_SAMPLES, dtype=int
        )
        signals_filtered = signals_filtered[:, depth_indices]
        depths_filtered = depths_filtered[depth_indices]
    elif DEPTH_DOWNSAMPLE_FACTOR > 1:
        depth_indices = np.arange(0, len(depths_filtered), DEPTH_DOWNSAMPLE_FACTOR)
        signals_filtered = signals_filtered[:, depth_indices]
        depths_filtered = depths_filtered[depth_indices]

    # 3. Downsample pings if needed
    if len(timestamps) > MAX_PINGS:
        ping_indices = np.linspace(0, len(timestamps) - 1, MAX_PINGS, dtype=int)
        signals_filtered = signals_filtered[ping_indices, :]
        timestamps_filtered = timestamps[ping_indices]
        d_top_true_filtered = d_top_true[ping_indices]
        d_top_gen_filtered = d_top_gen[ping_indices]
        d_bottom_true_filtered = d_bottom_true[ping_indices]
        d_bottom_gen_filtered = d_bottom_gen[ping_indices]
    elif PING_DOWNSAMPLE_FACTOR > 1:
        ping_indices = np.arange(0, len(timestamps), PING_DOWNSAMPLE_FACTOR)
        signals_filtered = signals_filtered[ping_indices, :]
        timestamps_filtered = timestamps[ping_indices]
        d_top_true_filtered = d_top_true[ping_indices]
        d_top_gen_filtered = d_top_gen[ping_indices]
        d_bottom_true_filtered = d_bottom_true[ping_indices]
        d_bottom_gen_filtered = d_bottom_gen[ping_indices]
    else:
        timestamps_filtered = timestamps
        d_top_true_filtered = d_top_true
        d_top_gen_filtered = d_top_gen
        d_bottom_true_filtered = d_bottom_true
        d_bottom_gen_filtered = d_bottom_gen

    # 4. Clip signal values
    signals_filtered = np.clip(signals_filtered, min_sv, max_sv)

    return (
        signals_filtered,
        timestamps_filtered,
        depths_filtered,
        d_top_true_filtered,
        d_top_gen_filtered,
        d_bottom_true_filtered,
        d_bottom_gen_filtered,
    )


@st.cache_data  # Cache data loading to speed up navigation
def load_data_for_file(
    f_path,
    max_pings,
    max_depth_samples,
    min_depth_meters,
    max_depth_meters,
    min_sv,
    max_sv,
):
    """
    Loads all necessary data for a single echogram file using echofilter data loaders.
    """
    try:
        # 1. Load echogram (signals, timestamps, depths)
        ts_raw, depths_raw, signals_raw = echofilter.raw.loader.transect_loader(
            f_path, row_len_selector="max"
        )

        # 2. Load ground truth top line
        true_evl_path = f_path.split(RAW_FILE_SUFFIX)[0] + TRUTH_TOP_FILE_SUFFIX
        _, d_top_true = echofilter.raw.loader.evl_loader(true_evl_path)

        # 3. Load generated top line
        gen_evl_path = f_path.split(".csv")[0] + GEN_TOP_FILE_PREFIX + MODEL_STR + ".evl"
        _, d_top_gen = echofilter.raw.loader.evl_loader(gen_evl_path)

        # Determine number of pings for interpolation and fallbacks
        num_pings = signals_raw.shape[0]

        # Interpolate top lines to match number of pings
        d_top_true = np.interp(
            np.linspace(0, len(d_top_true), num_pings),
            np.arange(len(d_top_true)),
            d_top_true,
        )
        d_top_gen = np.interp(
            np.linspace(0, len(d_top_gen), num_pings),
            np.arange(len(d_top_gen)),
            d_top_gen,
        )

        # 4. Load ground truth bottom line (suffix: _bottom.evl)
        bottom_truth_path = f_path.split(RAW_FILE_SUFFIX)[0] + TRUTH_BOT_FILE_SUFFIX
        try:
            _, d_bottom_true_raw = echofilter.raw.loader.evl_loader(bottom_truth_path)
            d_bottom_true = np.interp(
                np.linspace(0, len(d_bottom_true_raw), num_pings),
                np.arange(len(d_bottom_true_raw)),
                d_bottom_true_raw,
            )
        except FileNotFoundError:
            st.warning(f"Bottom truth EVL not found: {os.path.basename(bottom_truth_path)}")
            d_bottom_true = np.full(num_pings, np.nan)

        # 5. Load generated bottom line (suffix: _Sv_raw.bottom-<MODEL_STR>.evl)
        gen_bottom_path = f_path.split(".csv")[0] + GEN_BOT_FILE_PREFIX + MODEL_STR + ".evl"
        try:
            _, d_bottom_gen_raw = echofilter.raw.loader.evl_loader(gen_bottom_path)
            d_bottom_gen = np.interp(
                np.linspace(0, len(d_bottom_gen_raw), num_pings),
                np.arange(len(d_bottom_gen_raw)),
                d_bottom_gen_raw,
            )
        except FileNotFoundError:
            st.warning(f"Generated bottom EVL not found: {os.path.basename(gen_bottom_path)}")
            d_bottom_gen = np.full(num_pings, np.nan)
        # Reduce data size to avoid message size limits
        (
            signals_reduced,
            timestamps_reduced,
            depths_reduced,
            d_top_true_reduced,
            d_top_gen_reduced,
            d_bottom_true_reduced,
            d_bottom_gen_reduced,
        ) = reduce_data_size(
            signals_raw, ts_raw, depths_raw, d_top_true, d_top_gen, d_bottom_true, d_bottom_gen, min_sv, max_sv
        )

        return {
            "signals": signals_reduced,
            "timestamps": timestamps_reduced,
            "depths": depths_reduced,
            "d_top_true": d_top_true_reduced,
            "d_top_gen": d_top_gen_reduced,
            "d_bottom_true": d_bottom_true_reduced,
            "d_bottom_gen": d_bottom_gen_reduced,
            "original_shape": signals_raw.shape,  # Keep track of original size
        }
    except FileNotFoundError as e:
        st.error(f"Error loading files for {os.path.basename(f_path)}: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


def create_interactive_plot(data):
    """Creates an interactive Plotly figure from the loaded data."""
    fig = go.Figure()

    # EK500 colormap colors (from echopype_cm.py)
    ek500_colors = [
        [0, "rgb(159,159,159)"],  # light grey
        [0.1, "rgb(95,95,95)"],  # grey
        [0.2, "rgb(0,0,255)"],  # dark blue
        [0.3, "rgb(0,0,127)"],  # blue
        [0.4, "rgb(0,191,0)"],  # green
        [0.5, "rgb(0,127,0)"],  # dark green
        [0.6, "rgb(255,255,0)"],  # yellow
        [0.7, "rgb(255,127,0)"],  # orange
        [0.8, "rgb(255,0,191)"],  # pink
        [0.9, "rgb(255,0,0)"],  # red
        [1.0, "rgb(166,83,60)"],  # light brown
    ]

    # Convert timestamps to datetimes for readable x-axis
    x_dt = pd.to_datetime(data["timestamps"], unit="s", utc=True)

    # Add the echogram heatmap
    fig.add_trace(
        go.Heatmap(
            z=data["signals"].T,  # Transpose for correct orientation, already clipped
            x=x_dt,
            y=data["depths"],
            colorscale=ek500_colors,
            colorbar=dict(title="Signal (dB)", len=0.8, y=0.5),
        )
    )

    # Add the ground truth line
    fig.add_trace(
        go.Scatter(
            x=x_dt,
            y=data["d_top_true"],
            mode="lines",
            name="Ground Truth Top Line",
            line=dict(color="black", width=2),
        )
    )

    # Add the generated line
    fig.add_trace(
        go.Scatter(
            x=x_dt,
            y=data["d_top_gen"],
            mode="lines",
            name="Generated Top Line",
            line=dict(color="red", width=2),
        )
    )

    # Add the ground truth bottom line
    fig.add_trace(
        go.Scatter(
            x=x_dt,
            y=data["d_bottom_true"],
            mode="lines",
            name="Ground Truth Bottom Line",
            line=dict(color="black", width=2, dash="dot"),
        )
    )

    # Add the generated bottom line
    fig.add_trace(
        go.Scatter(
            x=x_dt,
            y=data["d_bottom_gen"],
            mode="lines",
            name="Generated Bottom Line",
            line=dict(color="red", width=2, dash="dot"),
        )
    )

    # Decide y-axis depth range
    full_max_depth = float(np.nanmax(data["depths"])) if len(data["depths"]) > 0 else None

    # If user enables manual max depth, respect it; otherwise use deepest bottom or full depth
    if 'USE_MANUAL_MAX_DEPTH' in globals() and USE_MANUAL_MAX_DEPTH and full_max_depth is not None:
        target_max_depth = min(float(MAX_DEPTH_METERS), full_max_depth)
    else:
        try:
            max_bottom_depth = np.nanmax(
                np.concatenate([
                    np.asarray(data.get("d_bottom_true", [])).ravel(),
                    np.asarray(data.get("d_bottom_gen", [])).ravel(),
                ])
            )
        except Exception:
            max_bottom_depth = np.nan
        # Add 1 m offset below the deepest bottom, cap by available depth
        if np.isfinite(max_bottom_depth) and full_max_depth is not None:
            target_max_depth = min(float(max_bottom_depth) + 1.0, full_max_depth)
        else:
            target_max_depth = full_max_depth

    # Update layout
    fig.update_layout(
        title="Echogram Inspector",
        xaxis_title="Time",
        yaxis_title="Depth (m)",
        yaxis=dict(
            autorange="reversed" if target_max_depth is None else False,
            range=[target_max_depth, float(MIN_DEPTH_METERS)] if target_max_depth is not None else None,
        ),  # Depths increase downwards
        height=800,  # Increase plot height
        margin=dict(l=80, r=120, t=110, b=60),  # Extra top margin for centered legend
        legend=dict(
            orientation="h",
            x=0.5,
            y=1.06,
            xanchor="center",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
        ),
    )

    # Ensure x-axis is treated as date and set a readable tick format
    fig.update_xaxes(type="date", tickformat="%Y-%m-%d\n%H:%M:%S")
    return fig


# --- STREAMLIT UI ---

st.set_page_config(page_title="Echogram Inspector", page_icon="🔎", layout="wide")
st.title("Interactive Echogram Inspector 🔎")

# Initialize session state for tracking file index
if "file_index" not in st.session_state:
    st.session_state.file_index = 0
if "filenames" not in st.session_state:
    st.session_state.filenames = []
if "data_path" not in st.session_state:
    st.session_state.data_path = ""

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    data_path = st.text_input(
        "Enter Path to Data Directory",
        placeholder="/mnt/scratch/echohub/s3/echo-jh/ErieEastBasin2024/",
        value=st.session_state.data_path,
    )

    if st.button("Use Default Path"):
        st.session_state.data_path = (
            "/mnt/scratch/echohub/s3/echo-jh/ErieEastBasin2024/"
        )
        st.rerun()

    # Update session state when path changes
    if data_path != st.session_state.data_path:
        st.session_state.data_path = data_path

    st.header("Data Reduction Settings")
    st.caption("Adjust these to reduce data size and avoid browser limits")

    max_pings = st.slider("Max Pings", 100, 2000, MAX_PINGS, 100)
    max_depth_samples = st.slider("Max Depth Samples", 100, 1000, MAX_DEPTH_SAMPLES, 50)
    min_depth = st.number_input("Min Depth (m)", 0.0, 50.0, MIN_DEPTH_METERS, 0.5)
    USE_MANUAL_MAX_DEPTH = st.checkbox("Use manual max depth", value=False)
    max_depth = st.number_input("Max Depth (m)", 1.0, 100.0, MAX_DEPTH_METERS, 0.5, disabled=not USE_MANUAL_MAX_DEPTH)

    st.header("Signal Range Settings")
    st.caption("Adjust the SV (volume backscatter) display range")

    min_sv = st.number_input("Min SV (dB)", -120.0, 0.0, float(SIGNAL_CLIP_MIN), 1.0)
    max_sv = st.number_input("Max SV (dB)", -120.0, 0.0, float(SIGNAL_CLIP_MAX), 1.0)

    # Update global settings
    MAX_PINGS = max_pings
    MAX_DEPTH_SAMPLES = max_depth_samples
    MIN_DEPTH_METERS = min_depth
    MAX_DEPTH_METERS = max_depth
    SIGNAL_CLIP_MIN = min_sv
    SIGNAL_CLIP_MAX = max_sv

    if st.session_state.data_path and os.path.isdir(st.session_state.data_path):
        # Scan directory and subdirectories for raw files
        st.session_state.filenames = []
        for root, dirs, files in os.walk(st.session_state.data_path):
            for file in files:
                if file.endswith(RAW_FILE_SUFFIX):
                    st.session_state.filenames.append(os.path.join(root, file))
        st.session_state.filenames = sorted(st.session_state.filenames)

        if not st.session_state.filenames:
            st.warning("No raw data files found in this directory.")
        else:
            st.success(f"✅ Found {len(st.session_state.filenames)} files")
            # Navigation buttons
            col1, col2 = st.columns(2)

            # Previous button
            prev_disabled = st.session_state.file_index <= 0
            if col1.button(
                "⬅️ Previous File", use_container_width=True, disabled=prev_disabled
            ):
                st.session_state.file_index -= 1
                st.rerun()

            # Next button
            next_disabled = (
                st.session_state.file_index >= len(st.session_state.filenames) - 1
            )
            if col2.button(
                "Next File ➡️", use_container_width=True, disabled=next_disabled
            ):
                st.session_state.file_index += 1
                st.rerun()

            st.write(
                f"Showing file **{st.session_state.file_index + 1}** of **{len(st.session_state.filenames)}**"
            )

            # File selector dropdown
            file_options = [os.path.basename(f) for f in st.session_state.filenames]

            selected_index = st.selectbox(
                "Select File:",
                options=range(len(file_options)),
                format_func=lambda x: file_options[x],
                index=st.session_state.file_index,
                key="file_dropdown",
            )

            # Update file index if dropdown selection changed
            if selected_index != st.session_state.file_index:
                st.session_state.file_index = selected_index
                st.rerun()

            st.info(f"Current file: **{file_options[st.session_state.file_index]}**")

            # Show dropdown status
            st.caption(f"Dropdown shows: {file_options[selected_index]}")

            # Refresh button
            if st.button("🔄 Refresh Data", use_container_width=True):
                # Only clear the data loading cache, preserve file list and UI state
                if "load_data_for_file" in st.cache_data._cached_funcs:
                    st.cache_data._cached_funcs["load_data_for_file"].clear()
                st.rerun()

    elif st.session_state.data_path:
        st.error("The provided path is not a valid directory.")

# Main content area
if st.session_state.filenames:
    current_file = st.session_state.filenames[st.session_state.file_index]

    # Debug info (can be removed later)
    with st.expander("Debug Info", expanded=False):
        st.write(
            f"File {st.session_state.file_index + 1} of {len(st.session_state.filenames)}"
        )
        st.write(f"Current: {os.path.basename(current_file)}")

    # Load data and create plot
    data = load_data_for_file(
        current_file,
        MAX_PINGS,
        MAX_DEPTH_SAMPLES,
        MIN_DEPTH_METERS,
        MAX_DEPTH_METERS,
        SIGNAL_CLIP_MIN,
        SIGNAL_CLIP_MAX,
    )

    if data:
        # Display data reduction info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Original Shape",
                f"{data['original_shape'][0]} × {data['original_shape'][1]}",
            )
        with col2:
            st.metric(
                "Displayed Shape",
                f"{data['signals'].shape[0]} × {data['signals'].shape[1]}",
            )
        with col3:
            reduction_factor = (
                data["original_shape"][0] * data["original_shape"][1]
            ) / (data["signals"].shape[0] * data["signals"].shape[1])
            st.metric("Reduction Factor", f"{reduction_factor:.1f}x")

        # Calculate MAE for display
        mae = np.mean(np.abs(data["d_top_true"] - data["d_top_gen"]))
        st.metric(label="Mean Absolute Error (m)", value=f"{mae:.4f}")

        # Create and display the plot
        fig = create_interactive_plot(data)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Enter a valid data directory path in the sidebar to begin.")
