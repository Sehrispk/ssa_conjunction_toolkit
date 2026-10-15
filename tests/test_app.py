"""Offline checks of the HTTP-to-CZML pipeline using a real SGP4 satellite."""

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.utils import iers
from sgp4.api import Satrec, jday

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "ssa-orbit-conjunction-toolkit"))
from app import app
from data_classes import MeanElements, OrbitData, SpaceObject


class ViewerAPITests(unittest.TestCase):
    def setUp(self):
        config = iers.conf.set_temp("auto_download", False)
        config.__enter__()
        self.addCleanup(config.__exit__, None, None, None)
        self.client = app.test_client()
        self.satrec = Satrec.twoline2rv(
            "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
            "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667",
        )
        elements = MeanElements(datetime(2000, 6, 27, tzinfo=timezone.utc),
                                10.82419157, 0.1859667, 34.2682,
                                348.7242, 331.7664, 19.3264)
        self.obj = SpaceObject("VANGUARD 1", "1958-002B", 5,
                               OrbitData(elements, "UTC", "TEME", "SGP4", "EARTH", None))
        self.payload = dict(objects=["VANGUARD 1"],
                            start_date="2000-06-28T18:00:00Z",
                            end_date="2000-06-28T20:00:00Z")

    def test_serves_viewer_and_script(self):
        for url, expected in (("/", b'cesiumContainer'),
                              ("/viewer/viewer.js", b'fetch("/propagate"')):
            with self.client.get(url) as response:
                self.assertEqual(response.status_code, 200)
                self.assertIn(expected, response.data)

    def test_propagation_returns_loadable_inertial_czml_in_metres(self):
        with patch("app.query", return_value=([self.obj], [self.satrec])):
            response = self.client.post("/propagate", json=self.payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/json")
        preamble, satellite = response.get_json()
        self.assertEqual(preamble["id"], "document")
        self.assertEqual(preamble["version"], "1.0")
        self.assertEqual(satellite["id"], "5")
        self.assertTrue(satellite["availability"].startswith(self.payload["start_date"][:-1]))
        position = satellite["position"]
        self.assertEqual(position["referenceFrame"], "INERTIAL")
        samples = np.array(position["cartesian"]).reshape(-1, 4)
        np.testing.assert_allclose(samples[:, 0], np.arange(0, 7201, 60))
        error, r_teme, _ = self.satrec.sgp4(*jday(2000, 6, 28, 18, 0, 0))
        self.assertEqual(error, 0)
        np.testing.assert_allclose(np.linalg.norm(samples[0, 1:]),
                                   np.linalg.norm(r_teme) * 1000, rtol=1e-10)
        self.assertGreater(np.linalg.norm(samples[0, 1:] / 1000 - r_teme), 0.1)

    def test_repeated_satellites_are_deduplicated(self):
        with patch("app.query", return_value=([self.obj], [self.satrec])):
            response = self.client.post("/propagate", json={
                **self.payload, "objects": ["VANGUARD 1", "VANGUARD"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()), 2)

    def test_satellite_metadata_is_in_response_and_click_description(self):
        with patch("app.query", return_value=([self.obj], [self.satrec])):
            response = self.client.post("/propagate", json=self.payload)
        self.assertEqual(response.status_code, 200)
        satellite = response.get_json()[1]
        properties = satellite["properties"]
        self.assertEqual(properties["norad_id"], 5)
        self.assertEqual(properties["international_designator"], "1958-002B")
        self.assertEqual(properties["inclination_deg"], 34.2682)
        self.assertEqual(properties["mean_motion_rev_per_day"], 10.82419157)
        self.assertEqual(properties["elements_epoch"], "2000-06-27T00:00:00+00:00")
        self.assertEqual(properties["elements_reference_frame"], "TEME")
        self.assertEqual(properties["position_reference_frame"], "GCRS (ECI)")
        description = satellite["description"]
        self.assertIn("<th>NORAD ID</th><td>5</td>", description)
        self.assertIn("<th>Inclination [deg]</th><td>34.2682</td>", description)
        self.assertIn("not instantaneous osculating elements", description)

    def test_metadata_text_is_escaped_in_html(self):
        self.obj.name = '<script>alert("test")</script>'
        with patch("app.query", return_value=([self.obj], [self.satrec])):
            response = self.client.post("/propagate", json=self.payload)
        satellite = response.get_json()[1]
        self.assertNotIn("<script>", satellite["description"])
        self.assertIn("&lt;script&gt;", satellite["description"])
        self.assertEqual(satellite["properties"]["name"], self.obj.name)

    def test_invalid_requests_rejected_before_query(self):
        changes = ({"objects": []}, {"objects": "ISS"},
                   {"end_date": "2001-06-28T20:00:00Z"}, {"start_date": "yesterday"},
                   {"start_date": "2000-06-29T18:00:00Z"},
                   {"start_date": "2000-06-28T18:00:00"})
        with patch("app.query") as query:
            for change in changes:
                with self.subTest(change=change):
                    response = self.client.post("/propagate", json={**self.payload, **change})
                    self.assertEqual(response.status_code, 400)
                    self.assertIn("error", response.get_json())
            query.assert_not_called()

    def test_minute_steps_omit_partial_last_minute(self):
        for end, expected in (("2000-06-28T18:02:30Z", [0, 60, 120]),
                              ("2000-06-28T18:00:30Z", [0, 30])):
            with self.subTest(end=end), patch("app.query", return_value=([self.obj], [self.satrec])):
                response = self.client.post("/propagate", json={**self.payload, "end_date": end})
                self.assertEqual(response.status_code, 200)
                positions = response.get_json()[1]["position"]["cartesian"]
                np.testing.assert_allclose(np.array(positions).reshape(-1, 4)[:, 0], expected)

    def test_missing_satellite_returns_json_error(self):
        with patch("app.query", return_value=([], [])):
            response = self.client.post("/propagate", json=self.payload)
        self.assertEqual(response.status_code, 404)
        self.assertIn("error", response.get_json())


if __name__ == "__main__":
    unittest.main()
