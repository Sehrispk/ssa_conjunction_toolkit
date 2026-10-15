"""Matplotlib views of propagated positions and Earth-centred mean elements.

Nothing here calls ``show()``. 3-D functions return ``(figure, axes, lines)``;
the element history and dashboard functions return ``(figure, axes)``.
Pass an existing ``ax`` to add orbits, then use the returned Line3D artists'
``set_color``, ``set_linestyle`` and ``set_linewidth`` methods to restyle them.
All positions/radii are in km and element angles are in degrees. Positions
overlaid on the same axes must use the TEME reference frame.
"""

from collections.abc import Mapping
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from sgp4.conveniences import dump_satrec
from sys import stdout

EARTH_RADIUS_KM = 6378.137

# Dataclass field, OMM field, axis title, unit.
_ELEMENTS = (
    ("mean_motion", "MEAN_MOTION", "Mean motion", "rev/day"),
    ("eccentricity", "ECCENTRICITY", "Eccentricity", ""),
    ("inclination", "INCLINATION", "Inclination", "deg"),
    ("raan", "RA_OF_ASC_NODE", "RAAN", "deg"),
    ("arg_of_pericenter", "ARG_OF_PERICENTER", "Argument of pericentre", "deg"),
    ("mean_anomaly", "MEAN_ANOMALY", "Mean anomaly", "deg"),
)


def _elements(record):
    """Accept MeanElements, OrbitData, SpaceObject, or a flat OMM mapping."""
    if hasattr(record, "orbit"):
        record = record.orbit
    if hasattr(record, "elements"):
        record = record.elements
    return record


def _value(record, field, omm_field):
    record = _elements(record)
    return record[omm_field] if isinstance(record, Mapping) else getattr(record, field)


def _records(value):
    if isinstance(value, Mapping) or hasattr(value, "epoch") or hasattr(value, "orbit") or hasattr(value, "elements"):
        return [value]
    return list(value)


def _labels(labels, count):
    if labels is None:
        return [None] * count
    labels = [labels] if isinstance(labels, str) else list(labels)
    if len(labels) != count:
        raise ValueError("Supply one label per orbit/trajectory.")
    return labels


def _orbit_axes(ax, earth_radius, show_earth):
    if not np.isfinite(earth_radius) or earth_radius <= 0:
        raise ValueError("earth_radius must be positive and finite (km).")
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
    if getattr(ax, "name", None) != "3d":
        raise ValueError("ax must be a Matplotlib 3-D axes.")
    if show_earth and not getattr(ax, "_ssa_earth_drawn", False):
        longitude = np.linspace(0, 2 * np.pi, 48)
        colatitude = np.linspace(0, np.pi, 24)
        x = earth_radius * np.outer(np.cos(longitude), np.sin(colatitude))
        y = earth_radius * np.outer(np.sin(longitude), np.sin(colatitude))
        z = earth_radius * np.outer(np.ones_like(longitude), np.cos(colatitude))
        ax.plot_surface(x, y, z, color="steelblue", alpha=0.35, linewidth=0)
        # Highlight the true equator.
        equator_angle = np.linspace(0, 2 * np.pi, 180)
        equator_radius = earth_radius * 1.002
        ax.plot(equator_radius * np.cos(equator_angle),
                equator_radius * np.sin(equator_angle), np.zeros_like(equator_angle),
                color="dimgray", linewidth=1, label="_equator")
        ax.text2D(0.02, 0.98, "XY: equatorial plane",
                  transform=ax.transAxes, fontsize=8, va="top",
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
        ax._ssa_earth_drawn = True
    ax.set_xlabel("Equatorial X [km]\nMean equinox direction", labelpad=14)
    ax.set_ylabel("Equatorial Y [km]\n90° from mean equinox", labelpad=14)
    ax.set_zlabel("North polar axis [km]", labelpad=10)
    return ax


def _equal_limits(ax, positions, earth_radius):
    # Preserve the extent of earlier additions and centre all views on Earth.
    extent = max(earth_radius, float(np.max(np.abs(positions)))) * 1.05
    extent = max(extent, getattr(ax, "_ssa_extent", 0))
    ax._ssa_extent = extent
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent))
    ax.set_box_aspect((1, 1, 1), zoom=0.8)


def plot_ephemeris(positions, *, ax=None, labels=None, earth_radius=EARTH_RADIUS_KM,
                   show_earth=True, show_start_position=True, show_last_position=True,
                   **line_kwargs):
    """Plot one ``(times, 3)`` or multiple ``(objects, times, 3)`` trajectories.

    Pass the ``r`` array returned by ``generate_ephemeris``. ``labels`` is a
    string for one trajectory or a sequence with one entry per object.
    Matplotlib line options (color, linestyle, linewidth, etc.) apply to this
    addition only. Return ``(fig, ax, lines)``; reuse ``ax`` for later additions.
    Dots labelled Start and End mark the first and final samples and follow
    the line colour. Set ``show_start_position=False`` or
    ``show_last_position=False`` to hide either endpoint. Marker/markersize
    options can customise their appearance.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 2:
        positions = positions[None, ...]
    if (positions.ndim != 3 or positions.shape[-1] != 3
            or positions.shape[0] == 0 or positions.shape[1] == 0):
        raise ValueError("positions must have shape (times, 3) or (objects, times, 3), with nonempty dimensions.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must contain only finite coordinates.")
    names = _labels(labels, len(positions))
    ax = _orbit_axes(ax, earth_radius, show_earth)
    lines = []
    for trajectory, label in zip(positions, names):
        options = dict(line_kwargs)
        endpoints = []
        if show_start_position:
            endpoints.append(0)
        if show_last_position and (len(trajectory) > 1 or not endpoints):
            endpoints.append(len(trajectory) - 1)
        if endpoints:
            options.setdefault("marker", "o")
            options.setdefault("markersize", 6)
            options.setdefault("markevery", endpoints)
        if label is not None:
            options["label"] = label
        lines.extend(ax.plot(*trajectory.T, **options))
        text_options = {"fontsize": 8, "bbox": {"facecolor": "white", "edgecolor": "none", "alpha": 0.8}}
        if len(trajectory) == 1 and show_start_position and show_last_position:
            ax.text(*trajectory[0], "Start / End", **text_options)
        else:
            if show_start_position:
                ax.text(*trajectory[0], "Start", **text_options)
            if show_last_position:
                ax.text(*trajectory[-1], "End", **text_options)
    _equal_limits(ax, positions, earth_radius)
    if any(name is not None for name in names) or line_kwargs.get("label"):
        ax.legend()
    return ax.figure, ax, lines


def _element_axes(host):
    """Six independent y scales on one shared time plot."""
    axes = np.array([host] + [host.twinx() for _ in range(5)], dtype=object)
    for index, ax in enumerate(axes):
        side = "left" if index < 3 else "right"
        ax.spines[side].set_position(("outward", 95 * (index % 3)))
        ax.yaxis.set_label_position(side)
        ax.yaxis.set_ticks_position(side)
        if index:
            ax.patch.set_visible(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.spines[side].set_visible(True)
    return axes


def plot_orbital_elements(history, *, times=None, axes=None, label=None,
                          unwrap_angles=False, **line_kwargs):
    """Plot all six mean elements together with six independent y-axes.

    ``history`` is a sequence of MeanElements/OrbitData/SpaceObject/OMM records
    for one object, or ``{label: history}`` for several objects. Epochs are used
    unless ``times`` is supplied (one sequence, or a matching label mapping).
    OMM ISO epoch strings are accepted. Pass the returned axes to add histories.
    Wrapped angular curves break at 360°/0° crossings rather than drawing a
    diagonal reset. Use ``unwrap_angles=True`` for continuous angular values.
    Return ``(fig, axes)`` with six overlaid axes in element order (mean motion,
    eccentricity, inclination, RAAN, argument of pericentre, mean anomaly).
    Each scale and curve has a matching colour and a label including units.
    ``axes`` can also be one 2-D axes to create the shared plot within a grid.
    Artists are accessible through each element axes' ``lines`` collection.
    """
    if isinstance(history, Mapping) and "MEAN_MOTION" not in history:
        series = list(history.items())
    else:
        series = [(label, history)]
    if not series:
        raise ValueError("Supply at least one element history.")
    prepared = []
    for name, records in series:
        records = _records(records)
        if not records:
            raise ValueError("Element histories must not be empty.")
        dates = times[name] if isinstance(times, Mapping) else times
        if dates is None:
            dates = [_value(record, "epoch", "EPOCH") for record in records]
        dates = [datetime.fromisoformat(date.replace("Z", "+00:00"))
                 if isinstance(date, str) else date for date in dates]
        if len(dates) != len(records):
            raise ValueError("Supply one time per element record.")
        values = np.array([[float(_value(record, field, omm))
                            for field, omm, _, _ in _ELEMENTS] for record in records])
        if not np.all(np.isfinite(values)):
            raise ValueError("Element histories must contain finite values.")
        if unwrap_angles:
            values[:, 3:] = np.rad2deg(np.unwrap(np.deg2rad(values[:, 3:]), axis=0))
        prepared.append((name, dates, values))
    if axes is None:
        fig, host = plt.subplots(figsize=(14, 7), layout="constrained")
        axes = _element_axes(host)
    elif hasattr(axes, "plot"):
        if axes.name == "3d":
            raise ValueError("Element axes must be 2-D.")
        fig = axes.figure
        axes = _element_axes(axes)
    else:
        axes = np.asarray(axes, dtype=object)
        if axes.size != 6:
            raise ValueError("axes must contain six Matplotlib axes.")
        axes = axes.reshape(6)
        fig = axes.flat[0].figure
        if any(ax.figure is not fig or ax.name == "3d" for ax in axes.flat):
            raise ValueError("Element axes must be 2-D and belong to one figure.")
        if any(not axes[0].get_shared_x_axes().joined(axes[0], ax) for ax in axes[1:]):
            raise ValueError("Element axes must share one time axis; pass axes returned by this function.")
    colours = plt.get_cmap("tab10").colors
    styles = ("-", "--", "-.", ":")
    for index, (ax, (_, _, title, unit)) in enumerate(zip(axes, _ELEMENTS)):
        for name, dates, values in prepared:
            options = dict(line_kwargs)
            options.setdefault("color", colours[index])
            options.setdefault("linestyle", styles[len(ax.lines) % len(styles)])
            options.setdefault("label", f"{name}: {title}" if name is not None else title)
            plot_dates = dates
            plot_values = values[:, index]
            if index >= 3 and not unwrap_angles:
                # Insert separators without discarding the samples on either
                # side of a 360°/0° crossing.
                breaks = np.flatnonzero(np.abs(np.diff(plot_values)) > 180) + 1
                if breaks.size:
                    date_array = np.asarray(dates, dtype=object)
                    plot_dates = np.insert(date_array, breaks, date_array[breaks])
                    plot_values = np.insert(plot_values, breaks, np.nan)
            ax.plot(plot_dates, plot_values, **options)
        colour = ax.lines[-1].get_color()
        ax.set_ylabel(f"{title} [{unit or 'dimensionless'}]", color=colour)
        ax.tick_params(axis="y", colors=colour)
        side = "left" if index < 3 else "right"
        ax.spines[side].set_color(colour)
        if index == 5 and not unwrap_angles:
            ax.set_ylim(0, 360)
    host = axes[0]
    host.set_title("Orbital elements")
    host.set_xlabel("Time [UTC]" if all(isinstance(date, datetime) for _, dates, _ in prepared for date in dates)
                    else "Time")
    host.grid(True, alpha=0.3)
    host.tick_params(axis="x", labelrotation=25)
    handles = [line for ax in axes for line in ax.lines if not line.get_label().startswith("_")]
    host.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=3)
    return fig, axes


def plot_orbit_dashboard(*, positions=None, element_history=None,
                         times=None, trajectory_labels=None,
                         ephemeris_style=None, elements_style=None):
    """Combine trajectories and the element timecourse in one subplot grid.

    Return ``(fig, axes)``. The axes dictionary contains ``'ephemeris'``,
    and/or ``'elements'`` (six overlaid axes), so each view
    can subsequently be extended with its individual plotting function.
    Style dictionaries accept the corresponding plotting function's options.
    """
    count = sum(value is not None for value in (positions, element_history))
    if count == 0:
        raise ValueError("Supply positions or element_history.")
    width = 14 if element_history is not None else 8
    fig = plt.figure(figsize=(width, 12 if count == 2 else 7), layout="constrained")
    grid = fig.add_gridspec(count, 1)
    axes = {}
    if positions is not None:
        ax = fig.add_subplot(grid[0, 0], projection="3d")
        plot_ephemeris(positions, ax=ax, labels=trajectory_labels, **(ephemeris_style or {}))
        ax.set_title("Propagated trajectories")
        axes["ephemeris"] = ax
    if element_history is not None:
        host = fig.add_subplot(grid[-1, :])
        _, element_axes = plot_orbital_elements(
            element_history, times=times, axes=host, **(elements_style or {}))
        axes["elements"] = element_axes
    return fig, axes


if __name__ == "__main__":
    from datetime import timedelta

    from sgp4.api import SGP4_ERRORS, SatrecArray, jday

    from data_classes import MeanElements
    from orbit_data_query import query
    from propagation import generate_ephemeris_array

    # Uses real CelesTrak OMM data, reusing the query cache when available.
    # ISS is the primary; China's Tianhe station is the secondary.
    space_objects, _ = query("STATIONS", query_type="GROUP")
    by_id = {obj.norad_id: obj for obj in space_objects}
    try:
        primary = by_id[25544]
        secondary = by_id[48274]
    except KeyError as exc:
        raise ValueError("The STATIONS data must contain ISS (25544) and Tianhe (48274).") from exc

    primary_label = f"{primary.name} (primary)"
    secondary_label = f"{secondary.name} (secondary)"
    # Propagate near the supplied epochs, so a cached set is also usable.
    start_time = max(primary.orbit.elements.epoch, secondary.orbit.elements.epoch)
    end_time = start_time + timedelta(hours=1)
    sample_count = 721  # 30-second spacing, including both endpoints.
    print(f"Propagating ISS and Tianhe from {start_time.isoformat()} to {end_time.isoformat()}")

    positions, velocities, dates = generate_ephemeris_array(
        SatrecArray([secondary.satrec]), [secondary], start_time, end_time, sample_count,
    )
    trajectory_fig, trajectory_ax, lines = plot_ephemeris(
        positions, labels=secondary_label, color="tab:blue", linewidth=1,
    )

    # Calculate and add the primary after plotting the secondary.
    primary_positions, primary_velocities, _ = generate_ephemeris_array(
        SatrecArray([primary.satrec]), [primary], start_time, end_time, sample_count,
    )
    _, _, primary_lines = plot_ephemeris(
        primary_positions, ax=trajectory_ax, labels=primary_label,
    )
    primary_lines[0].set(color="crimson", linestyle="--", linewidth=2.5)
    trajectory_ax.legend()
    trajectory_ax.set_title("SGP4 trajectories: ISS and Tianhe")

    # Snapshot SGP4's time-dependent singly averaged mean elements after each
    # scalar propagation. They are model predictions, not historical OMM sets.
    # Satrec angles are radians and mean motion is radians/minute.
    history = []
    for date in dates:
        jd, fraction = jday(date.year, date.month, date.day, date.hour, date.minute,
                            date.second + date.microsecond / 1e6)
        error, _, _ = primary.satrec.sgp4(jd, fraction)
        if error:
            raise ValueError(f"SGP4 failed for {primary.name} at {date}: {SGP4_ERRORS[error]}")
        history.append(MeanElements(
            epoch=date,
            mean_motion=primary.satrec.nm * 1440 / (2 * np.pi),
            eccentricity=primary.satrec.em,
            inclination=float(np.rad2deg(primary.satrec.im)),
            raan=float(np.rad2deg(primary.satrec.Om) % 360),
            arg_of_pericenter=float(np.rad2deg(primary.satrec.om) % 360),
            mean_anomaly=float(np.rad2deg(primary.satrec.mm) % 360),
        ))
    histories = {primary_label: history}

    element_fig, element_axes = plot_orbital_elements(histories)
    element_axes[0].set_title("ISS: SGP4 mean elements over time")

    dashboard_fig, dashboard_axes = plot_orbit_dashboard(
        positions=positions,
        trajectory_labels=secondary_label,
        element_history=histories,
    )
    plot_ephemeris(
        primary_positions, ax=dashboard_axes["ephemeris"], labels=primary_label,
        color="crimson", linestyle="--", linewidth=2.5,
    )
    dashboard_axes["elements"][0].set_title("ISS: SGP4 mean elements over time")
    dashboard_fig.suptitle(f"ISS and Tianhe — six hours from {start_time:%Y-%m-%d %H:%M UTC}")

    stdout.writelines(dump_satrec(primary.satrec))
    plt.show()
