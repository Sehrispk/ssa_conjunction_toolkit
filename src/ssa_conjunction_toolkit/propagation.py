from datetime import datetime, timedelta, timezone
from sys import stdout

import numpy as np
from sgp4.api import SGP4_ERRORS, Satrec, SatrecArray, jday
from sgp4.conveniences import dump_satrec

from .data_classes import SpaceObject
from .util import DATETIMES_2_JULIANS


def _validate_jd(julian_dates : np.ndarray, julian_date_fractions: np.ndarray):
    julian_dates = np.array(julian_dates)
    julian_date_fractions = np.array(julian_date_fractions)
    if len(julian_dates.shape) != 1 and len(julian_date_fractions.shape) != 1:
        raise ValueError("Invalid date format. Dates should be 1-dimensional arrays.")
    return julian_dates, julian_date_fractions

def _validate_error_messages(e : np.ndarray, space_objects: list[SpaceObject]):    
    if e.any():
        error_codes = e[e!=0]
        faulty_objects = np.array(space_objects)
        faulty_objects = faulty_objects[e[:,0]!=0]
        print(f"Error codes detected: {e[:,0]}")
        for code, object in zip(error_codes, faulty_objects):
            print(SGP4_ERRORS[code] + f" for space object {object.name} ({object.intdes}, {object.norad_id})")
            stdout.writelines(dump_satrec(object.satrec))

def _gen_date_range(start_date: datetime, end_date: datetime, N_dates: int) -> tuple[np.ndarray]:
    dates = np.array([start_date + (end_date - start_date) * t for t in np.linspace(0,1,N_dates)])
    jds = []
    jfrs = []
    jds, jfrs = DATETIMES_2_JULIANS(dates)
    return np.array(jds), np.array(jfrs), dates # (N_dates,) [float, float, datetime]

def propagate_satrecs(satrecs : SatrecArray, julian_dates : np.ndarray, julian_date_fractions: np.ndarray, space_objects : list[SpaceObject]) -> tuple[np.ndarray, np.ndarray]:
    julian_dates, julian_date_fractions = _validate_jd(julian_dates, julian_date_fractions)

    e, r, v = satrecs.sgp4(julian_dates, julian_date_fractions)

    _validate_error_messages(e, space_objects)

    return r, v # (N_sat, N_dates, 3) [km, km/s]

def propagate_satrec(satrec : Satrec,  julian_dates : np.ndarray, julian_date_fractions: np.ndarray, space_object : SpaceObject):
    julian_dates, julian_date_fractions = _validate_jd(julian_dates, julian_date_fractions)

    e, r, v = satrec.sgp4_array(julian_dates, julian_date_fractions)

    _validate_error_messages(e, [space_object])

    return r, v # (N_dates,3) 

def propagate_distance(jd : np.ndarray, jfr : np.ndarray, primary_satrec : Satrec, secondary_satrec : Satrec, primary_space_object, secondary_space_object) -> float:
    
    rp, _ = propagate_satrec(primary_satrec, jd, jfr, primary_space_object) # (N_dates, 3)
    rs, _ = propagate_satrec(secondary_satrec, jd, jfr, secondary_space_object) # (N_dates, 3)

    d = np.linalg.norm(np.array(rp)-np.array(rs), axis=1)  # (N_dates,)

    return d # km

def generate_ephemeris_array(satrecs : SatrecArray, space_objects: list[SpaceObject], start_date: datetime, end_date:datetime, N_dates : int):

    jds, jfrs, dates = _gen_date_range(start_date, end_date, N_dates)

    r, v = propagate_satrecs(satrecs, np.array(jds), np.array(jfrs), space_objects)

    return r, v, dates # d: (N_dates), r,v: (N_sat,N_dates,3) [km, km/s, datetime]

def generate_ephemeris(satrecs : Satrec, space_object: SpaceObject, start_date: datetime, end_date:datetime, N_dates : int):

    jds, jfrs, dates = _gen_date_range(start_date, end_date, N_dates)

    r, v = propagate_satrec(satrecs, np.array(jds), np.array(jfrs), space_object)

    return r, v, dates # d: (N_dates), r,v: (N_dates,3)

if __name__ == "__main__":
    from .orbit_data_query import query

    space_objects, satrecs = query("STATIONS", query_type="GROUP")

    jd, jfr = jday(2026, 10, 10, 12, 0, 0)
    r, v = propagate_satrecs(SatrecArray(satrecs), np.array([jd,]), np.array([jfr,]), space_objects)
    print(r.shape)
    print(v.shape)
    r, v = propagate_satrec(satrecs[0], np.array([jd,]), np.array([jfr,]), space_objects[0])
    print(r.shape)
    print(v.shape)

    # ephimeris example
    start_time = datetime.now(timezone.utc)
    end_time = datetime.now(timezone.utc)+timedelta(hours=2)
    dt = 2 # 2s interval
    N = int((end_time - start_time) / timedelta(seconds=2))
    r, v, d = generate_ephemeris_array(SatrecArray(satrecs), space_objects, start_time, end_time, N)
    print(r.shape)
    print(v.shape)
    print(d.shape)
    r, v, d = generate_ephemeris(satrecs[0], space_objects[0], start_time, end_time, N)
    print(r.shape)
    print(v.shape)

    from .util import TEME_2_GCRS
    r,v = TEME_2_GCRS(r,v,d)

    import czml3

    epoch=d[0]
    time_offsets = np.array([
        (date - epoch).total_seconds()
        for date in d
    ])

    traj_r = np.column_stack((time_offsets[:,None], r*1000)).flatten().tolist()

    packet = czml3.Packet(id=space_objects[0].name,
                          position=czml3.Position(cartesian=traj_r, epoch=epoch.isoformat().replace("+00:00", "Z")),
                          point=czml3.Point(show=True, color=czml3.Color(rgba=[0,0,255,255]), pixelSize=8),
                          )

    doc = czml3.Document(packets=[packet])

    filename = "example.czml"
    doc.save(filename)


