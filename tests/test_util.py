"""Offline regression checks for TEME state conversion."""

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy import coordinates as coord
from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.utils import iers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "ssa-orbit-conjunction-toolkit"))
from util import TEME_2_ICRF


class TEMEConversionTests(unittest.TestCase):
    def setUp(self):
        self.iers_config = iers.conf.set_temp("auto_download", False)
        self.iers_config.__enter__()
        self.addCleanup(self.iers_config.__exit__, None, None, None)
        self.r = np.array([7000.0, -1200.0, 500.0])
        self.v = np.array([1.0, 7.0, -2.0])
        self.date = Time("2024-01-01T00:00:00", scale="utc")

    def test_single_state_preserves_geocentric_distance_and_round_trips(self):
        r, v = TEME_2_ICRF(self.r.tolist(), self.v.tolist(), self.date)
        self.assertEqual(r.shape, (3,))
        self.assertEqual(v.shape, (3,))
        self.assertGreater(np.linalg.norm(r - self.r), 1.0)
        np.testing.assert_allclose(np.linalg.norm(r), np.linalg.norm(self.r), atol=1e-8)
        gcrs = coord.GCRS(
            coord.CartesianRepresentation(r * u.km).with_differentials(
                coord.CartesianDifferential(v * u.km / u.s)),
            obstime=self.date,
        )
        restored = gcrs.transform_to(coord.TEME(obstime=self.date))
        np.testing.assert_allclose(restored.cartesian.xyz.to_value(u.km), self.r, atol=1e-8)
        np.testing.assert_allclose(restored.velocity.d_xyz.to_value(u.km / u.s), self.v, atol=1e-8)

    def test_velocity_matches_position_derivative(self):
        # Move the TEME state and epoch together to check both the rotated
        # velocity and the contribution from the moving frame.
        step = 0.5
        before, _ = TEME_2_ICRF(self.r - step * self.v, self.v,
                               self.date - TimeDelta(step, format="sec"))
        after, _ = TEME_2_ICRF(self.r + step * self.v, self.v,
                              self.date + TimeDelta(step, format="sec"))
        _, velocity = TEME_2_ICRF(self.r, self.v, self.date)
        np.testing.assert_allclose(velocity, (after - before) / (2 * step),
                                   rtol=0, atol=1e-8)

    def test_inertial_frame_has_no_daily_earth_rotation(self):
        _, v = TEME_2_ICRF(self.r, np.zeros(3), self.date)
        # Only slow precession/nutation remains for a state stationary in
        # TEME; an Earth-fixed target would introduce about 0.5 km/s here.
        self.assertLess(np.linalg.norm(v), 1e-6)

    def test_trajectory_with_datetime_epochs(self):
        dates = np.array([datetime(2024, 1, 1, tzinfo=timezone.utc),
                          datetime(2024, 1, 2, tzinfo=timezone.utc)])
        positions = np.stack([self.r, self.r + 100])
        velocities = np.stack([self.v, self.v + 1])
        r, v = TEME_2_ICRF(positions, velocities, dates)
        self.assertEqual(r.shape, (2, 3))
        self.assertEqual(v.shape, (2, 3))
        for i, date in enumerate(dates):
            expected_r, expected_v = TEME_2_ICRF(positions[i], velocities[i], date)
            np.testing.assert_allclose(r[i], expected_r, atol=1e-8)
            np.testing.assert_allclose(v[i], expected_v, atol=1e-8)

    def test_multiple_objects_and_scalar_epoch(self):
        positions = np.broadcast_to(self.r, (2, 4, 3)).copy()
        positions[1] *= 2
        velocities = np.broadcast_to(self.v, positions.shape)
        for dates in (self.date, self.date + TimeDelta(np.arange(4), format="sec")):
            r, v = TEME_2_ICRF(positions, velocities, dates)
            self.assertEqual(r.shape, positions.shape)
            self.assertEqual(v.shape, velocities.shape)
            for i in range(2):
                expected_r, expected_v = TEME_2_ICRF(positions[i], velocities[i], dates)
                np.testing.assert_allclose(r[i], expected_r, atol=1e-8)
                np.testing.assert_allclose(v[i], expected_v, atol=1e-8)

    def test_invalid_shapes(self):
        for r, v in (([], []), ([1, 2], [1, 2]), (self.r, [1, 2]),
                     (np.zeros((2, 3)), self.v)):
            with self.assertRaises(ValueError):
                TEME_2_ICRF(r, v, self.date)
        for shape in ((2,), (1, 1)):
            with self.assertRaises(ValueError):
                TEME_2_ICRF(self.r, self.v, self.date + np.zeros(shape) * u.s)


if __name__ == "__main__":
    unittest.main()
