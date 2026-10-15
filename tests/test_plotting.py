"""Offline plotting and orbital geometry checks; run with unittest discovery."""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "ssa-orbit-conjunction-toolkit"))
from data_classes import MeanElements, OrbitData, SpaceObject
from plotting import (EARTH_MU_KM3_S2, orbit_from_elements, plot_ephemeris,
                      plot_omm_orbits, plot_orbital_elements, plot_orbit_dashboard)


class PlottingTests(unittest.TestCase):
    def setUp(self):
        self.elements = MeanElements(datetime(2026, 1, 1, tzinfo=timezone.utc),
                                     15.0, 0.1, 0.0, 0.0, 0.0, 359.0)

    def tearDown(self):
        plt.close("all")

    def test_ellipse_focus_closure_and_orientation(self):
        a = (EARTH_MU_KM3_S2 / (15 * 2 * np.pi / 86400) ** 2) ** (1 / 3)
        points = orbit_from_elements(self.elements)
        np.testing.assert_allclose(points[0], [a * 0.9, 0, 0], atol=1e-9)
        np.testing.assert_allclose(points[180], [-a * 1.1, 0, 0], atol=1e-9)
        np.testing.assert_allclose(points[0], points[-1], atol=1e-9)
        self.elements.inclination = 90
        polar = orbit_from_elements(self.elements)
        np.testing.assert_allclose(polar[:, 1], 0, atol=1e-9)
        self.assertGreater(polar[:, 2].max(), a * 0.9)
        self.elements.raan = 90
        np.testing.assert_allclose(orbit_from_elements(self.elements)[0],
                                   [0, a * 0.9, 0], atol=1e-9)

    def test_add_trajectories_and_restyle(self):
        points = orbit_from_elements(self.elements)
        fig, ax, lines = plot_ephemeris(np.stack([points, points * 2]), labels=["A", "B"])
        earth_count = len(ax.collections)
        previous_limit = ax.get_xlim()
        other_fig, other_ax, primary = plot_ephemeris(points, ax=ax, labels="Primary", linestyle="--")
        primary[0].set_color("red")
        primary[0].set_linewidth(3)
        self.assertIs(other_fig, fig)
        self.assertIs(other_ax, ax)
        self.assertEqual(len(lines), 2)
        self.assertEqual(primary[0].get_linewidth(), 3)
        self.assertEqual(primary[0].get_color(), "red")
        self.assertEqual(len(ax.collections), earth_count)
        self.assertEqual(ax.get_xlim(), previous_limit)
        self.assertEqual(len(ax.get_legend().texts), 3)
        self.assertEqual(ax.get_xlabel(), "Equatorial X [km]\nMean equinox direction")
        self.assertEqual(ax.get_ylabel(), "Equatorial Y [km]\n90° from mean equinox")
        self.assertEqual(ax.get_zlabel(), "North polar axis [km]")
        fig.canvas.draw()

    def test_omm_mapping_and_wrappers(self):
        omm = dict(EPOCH="2026-01-01T00:00:00Z", MEAN_MOTION="15",
                   ECCENTRICITY="0.1", INCLINATION="0", RA_OF_ASC_NODE="0",
                   ARG_OF_PERICENTER="0", MEAN_ANOMALY="359")
        orbit = OrbitData(self.elements, "UTC", "TEME", "SGP4", "EARTH", None)
        obj = SpaceObject("Example", "ID", 1, orbit)
        np.testing.assert_allclose(orbit_from_elements(omm), orbit_from_elements(obj))
        fig, ax, lines = plot_omm_orbits([obj, omm])
        self.assertEqual(lines[0].get_label(), "Example")
        self.assertEqual(len(lines), 2)
        fig.canvas.draw()
        fig, axes = plot_orbital_elements([omm])
        fig.canvas.draw()

    def test_histories_and_dashboard(self):
        later = MeanElements(self.elements.epoch + timedelta(days=1),
                             15.1, 0.11, 1, 1, 1, 1)
        history = [self.elements, later]
        fig, axes = plot_orbital_elements({"A": history}, unwrap_angles=True)
        np.testing.assert_allclose(axes[5].lines[0].get_ydata(), [359, 361])
        self.assertEqual(axes.shape, (6,))
        self.assertEqual(axes[0].get_ylabel(), "Mean motion [rev/day]")
        self.assertEqual(axes[1].get_ylabel(), "Eccentricity [dimensionless]")
        self.assertTrue(all("[deg]" in ax.get_ylabel() for ax in axes[2:]))
        self.assertEqual(axes[0].get_xlabel(), "Time [UTC]")
        self.assertTrue(all(axes[0].get_shared_x_axes().joined(axes[0], ax) for ax in axes[1:]))
        self.assertTrue(all(ax.get_position().bounds == axes[0].get_position().bounds for ax in axes))
        for ax in axes:
            self.assertEqual(ax.yaxis.label.get_color(), ax.lines[0].get_color())
        plot_orbital_elements(history, axes=axes, label="B")
        self.assertTrue(all(len(ax.lines) == 2 for ax in axes.flat))
        self.assertEqual(len(axes[0].get_legend().texts), 12)
        fig.canvas.draw()
        fig, views = plot_orbit_dashboard(positions=orbit_from_elements(self.elements),
                                         element_history=history, omm_elements=self.elements)
        self.assertEqual(set(views), {"ephemeris", "omm", "elements"})
        self.assertEqual(len(fig.axes), 8)
        fig.canvas.draw()
        fig, views = plot_orbit_dashboard(element_history=history)
        self.assertEqual(len(fig.axes), 6)
        fig.canvas.draw()

    def test_invalid_inputs(self):
        for positions in ([], np.zeros((3, 2)), [[1, 2, np.nan]]):
            with self.assertRaises(ValueError):
                plot_ephemeris(positions)
        with self.assertRaises(ValueError):
            plot_ephemeris(np.zeros((2, 3, 3)), labels=["only one"])
        with self.assertRaises(ValueError):
            plot_orbital_elements([self.elements], times=[])
        with self.assertRaises(ValueError):
            plot_orbit_dashboard()
        self.elements.eccentricity = 1
        with self.assertRaises(ValueError):
            orbit_from_elements(self.elements)


if __name__ == "__main__":
    unittest.main()
