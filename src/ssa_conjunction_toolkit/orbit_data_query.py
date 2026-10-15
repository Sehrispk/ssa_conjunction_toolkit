import datetime
import xml.etree.ElementTree as ET
from datetime import timezone
from pathlib import Path
import sgp4
from sgp4 import omm
from sgp4.api import Satrec, SatrecArray

import requests
from data_classes import *
import os
import time

DATA_BASE_URL="https://celestrak.org/NORAD/elements/gp.php"
DATA_REQUEST_FORMAT="&FORMAT=XML"

QUERY_TYPES=["ID", "GROUP", "NAME", "INTDES"]

SUPPORTED_TIME_SYSTEMS = ["UTC"]
SUPPORTED_REFERNCE_FRAMES = ["TEME"]
SUPPORTED_CENTERS = ["EARTH"]
SUPPORTED_GP_THEORY = ["SGP4"]

def _print_tree(element, level=0):
    indent = "  " * level

    print(f"{indent}{element.tag}: {element.text.strip() if element.text else ''}")

    for child in element:
        _print_tree(child, level + 1)

def _fetch_xml(query_string : str, query_type : str) -> tuple[ET, list[Satrec]]:
    # xml
    query_params = {query_type:query_string, "FORMAT":"XML"}
    cache_file_path = Path(f".query_cache_{query_type}_{query_string}.xml")
    if cache_file_path.exists() and time.time()-os.path.getmtime(cache_file_path) < 2*60*60:
        print("Loading cached orbital data")
        try:
            xml_response = ET.parse(cache_file_path).getroot()
        except Exception as e:
            raise ValueError("Failed to parse cached orbit data file.") from e
    else:
        response = requests.get(DATA_BASE_URL, params=query_params, timeout=15)
        response.raise_for_status()
        try:
            xml_response = ET.fromstring(response.content)
            cache_file_path.write_bytes(response.content)
        except Exception as e:
            print(f"XML parsing error: {e}")
            print(f"HTTP status: {response.status_code}")
            print(f"Content type: {response.headers.get('Content-Type')}")

    # omm parse
    satrecs = []
    with open(cache_file_path) as f:
        for fields in omm.parse_xml(f):
            satrec = Satrec()
            omm.initialize(satrec, fields)
            satrecs.append(satrec)

    return xml_response, satrecs
    
def _parse_celestrak_xml(xml_response : ET) -> list[SpaceObject]:
    space_objects = []
    for object_data in xml_response.findall(".//omm"):
        meta_data_xml = object_data.find("./body/segment/metadata")
        mean_elements_xml = object_data.find("./body/segment/data/meanElements")
        tle_parameters_xml = object_data.find("./body/segment/data/tleParameters")

        # manual parse
        # parse meta data
        time_system = meta_data_xml.findtext("TIME_SYSTEM")
        ref_frame = meta_data_xml.findtext("REF_FRAME")
        gp_theory = meta_data_xml.findtext("MEAN_ELEMENT_THEORY")
        object_name = meta_data_xml.findtext("OBJECT_NAME")
        object_id = meta_data_xml.findtext("OBJECT_ID")
        norad_id = int(tle_parameters_xml.findtext("NORAD_CAT_ID"))
        center_name = meta_data_xml.findtext("CENTER_NAME")
        if time_system not in SUPPORTED_TIME_SYSTEMS:
            raise ValueError(f"Celestrak response with {time_system} time system is not supported.")
        if ref_frame not in SUPPORTED_REFERNCE_FRAMES:
            raise ValueError(f"Celestrak response with {ref_frame} reference frame is not supported.")
        if center_name not in SUPPORTED_CENTERS:
            raise ValueError(f"Celestrak response with {center_name} center is not supported.")
        if gp_theory not in SUPPORTED_GP_THEORY:
            raise ValueError(f"Celestrak response with {gp_theory} gp theory is not supported.")


        # parse orbit data
        mean_elements = MeanElements(epoch=datetime.datetime.strptime(mean_elements_xml.findtext("EPOCH"), "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc),
                                     mean_motion=float(mean_elements_xml.findtext("MEAN_MOTION")),
                                     eccentricity=float(mean_elements_xml.findtext("ECCENTRICITY")),
                                     inclination = float(mean_elements_xml.findtext("INCLINATION")),
                                     raan=float(mean_elements_xml.findtext("RA_OF_ASC_NODE")),
                                     arg_of_pericenter=float(mean_elements_xml.findtext("ARG_OF_PERICENTER")),
                                     mean_anomaly=float(mean_elements_xml.findtext("MEAN_ANOMALY")))
        tle_parameters = TLEParameters(ephemeris_type=int(tle_parameters_xml.findtext("EPHEMERIS_TYPE")),
                                       classification_type=tle_parameters_xml.findtext("CLASSIFICATION_TYPE"),
                                       norad_cat_id=norad_id,
                                       element_set_no=int(tle_parameters_xml.findtext("ELEMENT_SET_NO")),
                                       rev_at_epoch=int(tle_parameters_xml.findtext("REV_AT_EPOCH")),
                                       bstar=float(tle_parameters_xml.findtext("BSTAR")),
                                       mean_motion_dot=float(tle_parameters_xml.findtext("MEAN_MOTION_DOT")),
                                       mean_motion_ddot=float(tle_parameters_xml.findtext("MEAN_MOTION_DDOT")))
        orbit_data = OrbitData(elements=mean_elements, tle_parameters=tle_parameters, frame=ref_frame, time_system=time_system, gp_theory=gp_theory, center_name=center_name)

        space_object = SpaceObject(name=object_name, intdes=object_id, norad_id=norad_id, orbit=orbit_data)
        space_objects.append(space_object)
    return space_objects


def query(query_string : str, query_type : str) -> (list[SpaceObject], list[Satrec]):
    # check input
    if query_type not in QUERY_TYPES:
        raise ValueError("Invalid query type.")

    xml_response, satrecs = _fetch_xml(query_string=query_string, query_type=query_type)

    space_objects = _parse_celestrak_xml(xml_response)
    for object, satrec in zip(space_objects, satrecs):
        object.satrec = satrec

    return space_objects, satrecs
    

if __name__=="__main__":
    query("STATIONS", query_type="GROUP")