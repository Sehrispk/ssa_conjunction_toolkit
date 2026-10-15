from html import escape

import czml3
import numpy as np
from astropy import coordinates as coord
from astropy import units as u
from astropy.time import Time


def _satellite_metadata(space_object):
    """Return machine-readable CZML properties and the selection-panel table."""
    orbit = space_object.orbit
    elements = orbit.elements
    fields = [
        ("name", "Name", space_object.name),
        ("norad_id", "NORAD ID", space_object.norad_id),
        ("international_designator", "International designator", space_object.intdes),
        ("center", "Orbit centre", orbit.center_name),
        ("propagation_theory", "Propagation theory", orbit.gp_theory),
        ("position_reference_frame", "Position reference frame", "GCRS (ECI)"),
        ("elements_reference_frame", "Mean elements reference frame", orbit.frame),
        ("elements_time_system", "Elements time system", orbit.time_system),
        ("elements_epoch", "Mean elements epoch", elements.epoch.isoformat()),
        ("mean_motion_rev_per_day", "Mean motion [rev/day]", elements.mean_motion),
        ("eccentricity", "Eccentricity", elements.eccentricity),
        ("inclination_deg", "Inclination [deg]", elements.inclination),
        ("raan_deg", "RAAN [deg]", elements.raan),
        ("arg_of_pericenter_deg", "Argument of pericentre [deg]", elements.arg_of_pericenter),
        ("mean_anomaly_deg", "Mean anomaly [deg]", elements.mean_anomaly),
    ]
    properties = {key: value for key, _, value in fields}
    rows = "".join(
        f"<tr><th>{escape(label)}</th><td>{escape(str(value))}</td></tr>"
        for _, label, value in fields
    )
    description = (
        '<table class="cesium-infoBox-defaultTable"><tbody>'
        + rows
        + "</tbody></table><p>Mean elements are from the source element set at "
        "the listed epoch, not instantaneous osculating elements.</p>"
    )
    return properties, description


def generate_czml_traj_packet(r, v, d, space_object):
    """Build a CZML packet from GCRS positions in km (Cesium uses metres)."""

    epoch=d[0]
    time_offsets = np.array([
        (date - epoch).total_seconds()
        for date in d
    ])

    traj_r = np.column_stack((time_offsets[:,None], r*1000)).flatten().tolist()
    properties, description = _satellite_metadata(space_object)

    packet = czml3.Packet(id=str(space_object.norad_id), name=space_object.name,
                          properties=properties, description=description,
                          availability=f"{epoch.isoformat()}/{d[-1].isoformat()}",
                          position=czml3.Position(cartesian=traj_r, epoch=epoch.isoformat().replace("+00:00", "Z"), referenceFrame="INERTIAL"),
                          point=czml3.Point(show=True, color=czml3.Color(rgba=[0,0,255,255]), pixelSize=8),
                          path=czml3.Path(show=True, width=1),
                          )

    return packet

def generate_czml_trajectory_packets(rs, vs, d, space_objects):
    packets = []
    for r, v, obj in zip(rs, vs, space_objects):
        packets.append(generate_czml_traj_packet(r, v, d, obj))
    return packets

def TEME_2_ICRF(r, v, d):

    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    obstime = d if isinstance(d, Time) else Time(d, scale="utc")

    rep = coord.CartesianRepresentation(r * u.km, xyz_axis=-1)
    diff = coord.CartesianDifferential(v * u.km / u.s, xyz_axis=-1)
    teme_coords = coord.TEME(rep.with_differentials(diff), obstime=obstime)
    gcrs_coords = teme_coords.transform_to(coord.GCRS(obstime=obstime))

    return (
        np.moveaxis(gcrs_coords.cartesian.xyz.to_value(u.km), 0, -1),
        np.moveaxis(gcrs_coords.velocity.d_xyz.to_value(u.km / u.s), 0, -1),
    )
