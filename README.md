# ssa-orbit-conjunction-toolkit
A small project for screening publicly available orbital elemts of space objects and screening for possbile conjunctions with regard to a primary satellite orbit.

## Cesium viewer

Run the Flask server from the repository root:

```sh
.venv/bin/python src/ssa-orbit-conjunction-toolkit/app.py
```

Open http://127.0.0.1:5000/, enter satellite names and a time range, and click
**Load trajectories**. Flask serves both the viewer and the API; opening the
HTML directly or on a separate static server will not connect to the API.
The globe uses Cesium's default imagery, imagery picker, and address search.
CesiumJS assets load from the CDN; the imagery and search services use Cesium's
default service configuration. Python needs Flask, NumPy, SGP4,
Requests, Astropy, and czml3. Element queries use CelesTrak or the local query
cache; keep cached elements current for the requested time range.

The browser sends a JSON POST to `/propagate`:

```json
{
  "objects": ["ISS (ZARYA)", "Tianhe"],
  "start_date": "2026-10-15T12:00:00Z",
  "end_date": "2026-10-15T14:00:00Z"
}
```

The API calculates the sample count from the requested duration, using one
sample per minute and omitting any partial final minute. A two-hour interval
produces 121 samples per satellite. Ranges shorter than one minute use their
two endpoints. The propagation functions retain their existing sample-count API.

The route propagates in TEME, converts to GCRS with `TEME_2_ICRF`, converts km
to metres, and returns a CZML array beginning with its document packet.
Positions use `referenceFrame: "INERTIAL"`; each sample contains seconds from
the epoch followed by X, Y, Z. `viewer/viewer.js` fetches that JSON and passes
it to `Cesium.CzmlDataSource.load`, then sets the clock to the sample interval
and adds a moving name label beside each satellite's point. Each name gets a
distinct color shared by its point and orbit, retained across reloads within
the same page session. Loading trajectories preserves the camera position.

Click a satellite's point, label, or orbit to open Cesium's information panel.
Each CZML packet includes a `description` HTML table with the NORAD ID,
international designator, source mean orbital elements, their epoch and units,
and the source and position reference frames. The same values are available
as typed CZML `properties` (for example `norad_id` and `inclination_deg`).
These are the source mean elements, not instantaneous osculating elements
calculated at the animation time. Extend `_satellite_metadata` in `util.py`
to add more fields to both the table and the response properties.

The existing `/conjunction` route is still unfinished: it needs HTTP request
parsing and CDM serialization. CDM event details belong in a report or selected
satellite's description. Its RTN tracks are relative offsets and must be
converted to absolute positions before they can be drawn on the globe.

Run the offline API checks with:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_app.py'
```

## Plotting

Run the offline demo with `.venv/bin/python src/ssa-orbit-conjunction-toolkit/plotting.py`.
It opens the individual plots and the combined dashboard using synthetic data,
including adding and styling a primary trajectory after the initial plot.

Install NumPy and Matplotlib in your Python environment. With
`src/ssa-orbit-conjunction-toolkit` on `PYTHONPATH`, import the plotting functions:

```python
import matplotlib.pyplot as plt
from plotting import (
    plot_ephemeris, plot_orbital_elements, plot_omm_orbits, plot_orbit_dashboard,
)

# r is the position array from generate_ephemeris: (objects, times, 3), in km.
fig, ax, lines = plot_ephemeris(r, labels=[obj.name for obj in space_objects])

# Calculate the primary satellite's ephemeris separately, then add it.
_, _, primary_lines = plot_ephemeris(
    primary_r, ax=ax, labels="Primary", color="red", linestyle="--", linewidth=2,
)
primary_lines[0].set_linewidth(3)
lines[0].set_color("gray")
fig.savefig("trajectories.png", dpi=150)

# A history is a sequence of MeanElements, OrbitData, SpaceObject, or flat
# OMM dictionaries, with an epoch per record. Compare objects via a mapping.
fig_elements, element_axes = plot_orbital_elements(
    {"Primary": primary_history, "Secondary": secondary_history},
)
# Add another history using the six y-axes on the shared time plot.
plot_orbital_elements(other_history, axes=element_axes, label="Other")

# Draw closed ellipses from the latest element sets (no propagation needed).
fig_orbits, orbit_ax, orbit_lines = plot_omm_orbits(space_objects)

# Combine all three views. Any of the three inputs can be omitted.
fig_grid, axes = plot_orbit_dashboard(
    positions=r,
    trajectory_labels=[obj.name for obj in space_objects],
    omm_elements=space_objects,
    element_history={"Primary": primary_history},
    ephemeris_style={"linewidth": 0.8},
)
plot_ephemeris(primary_r, ax=axes["ephemeris"], color="red", labels="Primary")
plt.show()
```

The 3-D functions return `(figure, axes, lines)`; each returned line supports
Matplotlib styling methods. The history function returns `(figure, axes)` with
six overlaid axes in a one-dimensional array, one y-axis for each element.
Curves, ticks, and axis labels use matching colours; each y-axis includes its
units (rev/day, degrees, or dimensionless). The dashboard returns `(figure, axes_dict)` with keys
`ephemeris`, `omm`, and `elements` for the views requested. All figures can be
saved with `figure.savefig(...)`; functions never call `show()` themselves.

Earth is a sphere at the origin. Use positions in kilometres in the same
TEME frame for all trajectories on an axis (SGP4 returns TEME). The 3-D axes
describe X as the equatorial mean equinox direction, Y as the equatorial
direction 90° from it, and Z as the north polar axis, all in kilometres.
The Earth model highlights the equator. Earth's orbital motion around the Sun
is not a fixed axis of this Earth-centred frame.
OMM ellipses use mean motion to infer semi-major axis and the angular elements
to orient the orbit, with Earth at a focus. They illustrate two-body geometry
from mean elements rather than the SGP4 path. Mean anomaly does not affect the
closed ellipse. Only bound elliptical orbits are supported.

Element histories plot mean motion, eccentricity, inclination, RAAN, argument
of pericentre, and mean anomaly against each record's epoch. Supply `times=`
to override the epochs; for multiple histories it may be a mapping from label
to time sequence. Use `unwrap_angles=True` to make RAAN, argument of pericentre,
and mean anomaly continuous across 360-degree boundaries. These functions
visualize supplied elements; they do not convert state vectors into elements.
