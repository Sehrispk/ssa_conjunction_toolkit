from html import escape
from datetime import datetime, timezone
from sgp4.conveniences import jday

import czml3
import numpy as np
from astropy import coordinates as coord
from astropy import units as u
from astropy.time import Time
import juliandate


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

def TEME_2_GCRS(rs, vs, d):
    """
        Transforms positions rs and velocities vs of object into the GCRS reference quasi intertial reference frame
        rs: (N_dates, 3)
        vs: (N_dates, 3)
        ds: (N_dates,)

        r_gcrs: (N_dates,3)
        v_gcrs: (N_dates,3)
    """

    rs = np.asarray(rs, dtype=float)
    vs = np.asarray(vs, dtype=float)

    obstime = d if isinstance(d, Time) else Time(d, scale="utc")

    rep = coord.CartesianRepresentation(rs * u.km, xyz_axis=-1)
    diff = coord.CartesianDifferential(vs * u.km / u.s, xyz_axis=-1)
    teme_coords = coord.TEME(rep.with_differentials(diff), obstime=obstime)
    gcrs_coords = teme_coords.transform_to(coord.GCRS(obstime=obstime)) # (3, N_dates)

    return (
        np.moveaxis(gcrs_coords.cartesian.xyz.to_value(u.km), 0, -1),
        np.moveaxis(gcrs_coords.velocity.d_xyz.to_value(u.km / u.s), 0, -1),
    )

def TEME_2_RTN(rp, vp, rs, vs):
    """
        Transforms positions rs and velocities vs of object into the RTN reference frame defined/anchored by the primary objects positions rp and velocities vp
        The transformation is anchored to rp, so each time point receives a new transformation matrix
        rp: (N_dates, 3)
        vp: (N_dates, 3)
        rs: (N_dates,3)
        vs: (N_dates,3)

        r_rtn: (N_dates,3)
        v_rtn: (N_dates,3)
    """

    N_dates = rp.shape[0]

    # calculate transforamtion matrix to RTN based on rp and vp
    e_r = rp/np.broadcast_to(np.linalg.norm(rp[...,None], axis=1), shape=(N_dates,3))
    h = np.cross(rp, vp, axisa=1, axisb=1)
    e_n = h / np.broadcast_to(np.linalg.norm(h[...,None], axis=1), shape=(N_dates,3))
    e_t = np.cross(e_n,e_r, axisa=1, axisb=1)
    M_rtn = np.transpose(np.array((e_r,e_t,e_n)), axes=(1,0,2))

    # calculate non-intertial frame effekt (scheinkraft) for velocity transform 
    omega = np.zeros_like(rp)
    omega[:,2] = np.linalg.norm(h, axis=1) / np.linalg.norm(rp, axis=1)**2

    # transform positions and veloccities
    r_rtn = np.einsum("tev,tv->te", M_rtn, (rs - rp))
    v_rtn = np.einsum("tev,tv->te", M_rtn, (vs - vp)) - np.cross(omega, r_rtn, axisa=1, axisb=1)

    return r_rtn, v_rtn

def JULIAN_2_DATETIME(jd: list[float], jfr: list[float]):
    """
        converts a julian date into a datetime utc format
    """
    return datetime(*juliandate.to_gregorian(jd+jfr), tzinfo=timezone.utc)

def JULIANS_2_DATETIMES(jds, jfrs):
    dates = []
    for jd, jfr in zip(jds, jfrs):
        dates.append(JULIAN_2_DATETIME(jd,jfr))
    return dates

def DATETIME_2_JULIAN(d: datetime) -> tuple[float]:
    """
        converts a datetime (utc) into julian date and fraction
    """
    jd, jfr = jday(
                d.year,
                d.month,
                d.day,
                d.hour,
                d.minute,
                d.second + d.microsecond / 1e6)
    return jd, jfr

def DATETIMES_2_JULIANS(ds: list[datetime]) -> tuple[list[float]]:   
    """
        converts a list/array of datetimes (utc) into julian dates and fractions
    """
    jds = []
    jfrs = []
    for date in ds:
        jd, jfr = DATETIME_2_JULIAN(date)
        jds.append(jd)
        jfrs.append(jfr)
    return jds, jfrs