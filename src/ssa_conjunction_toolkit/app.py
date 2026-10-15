from flask import Flask, Response, jsonify, request
from pathlib import Path
from orbit_data_query import query
from propagation import generate_ephemeris_array
import datetime
from datetime import timezone
from sgp4.api import SatrecArray
from util import TEME_2_ICRF, generate_czml_trajectory_packets
import czml3
import requests

VIEWER_DIR = Path(__file__).resolve().parents[2] / "viewer"
app = Flask(__name__, static_folder=str(VIEWER_DIR), static_url_path="/viewer")


@app.get("/")
def viewer():
    return app.send_static_file("index.html")


@app.post("/propagate")
def call_propagate():
    payload = request.get_json()
    try:
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        objects = payload["objects"]
        if not isinstance(objects, list) or not objects or not all(
            isinstance(name, str) and name.strip() for name in objects
        ):
            raise ValueError("objects must be a nonempty list of satellite names")
        start = datetime.datetime.fromisoformat(payload["start_date"].replace("Z", "+00:00"))
        end = datetime.datetime.fromisoformat(payload["end_date"].replace("Z", "+00:00"))
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("Dates must include a timezone, for example Z for UTC")
        start, end = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
        if end <= start:
            raise ValueError("end_date must be after start_date")
        full_minutes = int((end - start).total_seconds() // 60)
        samples = max(2, full_minutes + 1)
        if samples > 10000:
            raise ValueError("Time range exceeds 10000 samples at one-minute steps")
        if full_minutes:
            end = start + datetime.timedelta(minutes=full_minutes)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        return jsonify(error=f"Invalid propagation request: {error}"), 400

    space_objects, satrecs = [], []
    seen = set()
    try:
        for object_str in objects:
            found_objects, found_satrecs = query(object_str, query_type="NAME")
            if not found_objects:
                return jsonify(error=f"No satellites found for {object_str}"), 404
            for obj, satrec in zip(found_objects, found_satrecs):
                if obj.norad_id not in seen:
                    seen.add(obj.norad_id)
                    space_objects.append(obj)
                    satrecs.append(satrec)
    except requests.RequestException:
        return jsonify(error="Could not fetch orbital elements from CelesTrak"), 502

    rs, vs, d = generate_ephemeris_array(
        SatrecArray(satrecs), space_objects, start, end, samples
    )
    rs, vs = TEME_2_ICRF(rs, vs, d)
    packets = generate_czml_trajectory_packets(rs, vs, d, space_objects)
    preamble = czml3.Packet(id="document", name="Satellite trajectories", version="1.0")
    doc = czml3.Document(packets=[preamble, *packets])
    return Response(doc.dumps(), mimetype="application/json")

@app.route("/conjunction")
def call_conjunction_screening(primary_object: str, sceondary_obejcts: list[str]):
    from conjunction_analysis import conjunction_screening

    # query primary
    primary_space_objects, primary_satrecs = query(primary_object, query_type="NAME") 
    primary_satrec = primary_satrecs[0]
    primary_space_object = primary_space_objects[0]
    # query secondary
    secondary_space_objects,secondary_satrecs = query("Tianhe", query_type="NAME")
    # remove primary from secondary if needed
    for so in secondary_space_objects:
        if so.norad_id == primary_space_object.norad_id:
            secondary_space_objects.remove(so)

    cdms, cdm_tracks = conjunction_screening(primary_space_object, secondary_space_objects, primary_satrec, secondary_satrecs, start_time=datetime.datetime.now(tz=timezone.utc), screen_interval=4*24)

    return cdms


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)

