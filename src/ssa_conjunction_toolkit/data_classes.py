import datetime
from dataclasses import dataclass
from sgp4.api import Satrec


@dataclass
class MeanElements:
    epoch : datetime.datetime # UTC datetime
    mean_motion : float # rev / d
    eccentricity: float # no unit
    inclination: float # deg
    raan: float # deg
    arg_of_pericenter: float # deg
    mean_anomaly: float # deg

@dataclass
class TLEParameters:
    ephemeris_type : int
    classification_type : str
    norad_cat_id: int # norad catalog number
    element_set_no: int
    rev_at_epoch: int
    bstar: float
    mean_motion_dot: float # rev / d**2
    mean_motion_ddot: float # rev / d**3

@dataclass
class OrbitData:
    elements: MeanElements
    time_system: str
    frame: str
    gp_theory : str
    center_name: str # name of orbit center (ie. EARTH)
    tle_parameters: TLEParameters


@dataclass
class SpaceObject:
    # simple parsing/clone of celestrac data plus ephemeris and satrec fields. Mainly used for easy access.
    name: str # name of space object
    intdes: str # space object INTDES ID
    norad_id: int # norad catalog ID
    orbit: OrbitData # orbital elements retreived from celestrak
    ephemeris: list[OrbitData] | None = None # optional projected trajectory
    satrec : Satrec | None = None
