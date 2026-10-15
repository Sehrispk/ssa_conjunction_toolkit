from datetime import UTC, datetime, timedelta, timezone

import juliandate
import matplotlib.pyplot as plt
import numpy as np
import scipy
from .CDM.CDM import CDMT
from .data_classes import SpaceObject
from .orbit_data_query import query
from .propagation import (
    generate_ephemeris,
    generate_ephemeris_array,
    propagate_distance,
    propagate_satrec,
)
from sgp4.api import Satrec, SatrecArray
from sgp4.conveniences import jday


def L_distance_sum(julian_date_array : np.ndarray, primary_satrec: Satrec, secondary_satrec: Satrec, primary_space_object : SpaceObject, secondary_space_object : SpaceObject, target_distance : float = 0) -> float:
    """
        julian_date_array: (2*N_dates) -> (jd0,jfr0,jd1,jfr1,...)
    """
    jd = julian_date_array[0::2]
    jfr =julian_date_array[1::2]

    d = propagate_distance(jd, jfr, primary_satrec, secondary_satrec, primary_space_object, secondary_space_object) # (N_dates,)

    loss = np.sum(np.abs(d-target_distance)) # minimize distance sum to minimize individual distances

    return loss

def _find_distance_minima_in_ephemeris(rp : np.ndarray, rs : np.ndarray):
    # calculate index position of minimas of the distance d
    delta_r = np.broadcast_to(rp, (rs.shape[0], rs.shape[1], 3))-rs # (N_sat, N_t, 3) [km]
    d = np.linalg.norm(delta_r, axis=2) # (N_sat, N_t)
    d_dot = np.diff(d, axis=1) # (N_sat, N_t)
    mask = (d_dot[:, :-1] < 0) & (d_dot[:, 1:] > 0)
    minimas = [np.where(mask[i])[0] + 1 for i in range(d.shape[0])] # N_sat x (N_minima_i,)

    return minimas

def linear_distance_minimum_estimate(rp : np.ndarray, rs : np.ndarray, dates: np.ndarray[datetime]) -> tuple[np.ndarray,np.ndarray]:
    """
        estimates the minima of the distance between a primary objects and a set of secondary objects from the ephimeris positions of the objects and a linear model.
        rp, rs, dates -> t_mins, d_mins
        rp: (1, N_dates, 3)
        rs: (N_sat, N_dates, 3)
        dates: (N_dates,)

        t_mins: list(ndarray(N_min_i,))
        d_mins: list(ndarray(N_min_i,))
    """
    # calculate index position of minimas of the distance d
    minimas = _find_distance_minima_in_ephemeris(rp, rs) # N_sat x (N_minima_i,)
    delta_r = np.broadcast_to(rp, (rs.shape[0], rs.shape[1], 3))-rs # (N_sat, N_t, 3) [km]
    dt = float((dates[-1]-dates[0]).total_seconds())/dates.shape[0] # assume even spacing

    # for each secondary object use a linear approach to estimate the minimum distance and time in interval [t_min-1,t_min+1]    
    t_mins = []
    d_mins = []
    for i, minima in enumerate(minimas):
        d1 = delta_r[i, minima-1, :]  # (N_minima, 3) [km]
        d2 = delta_r[i, minima+1, :] # (N_minima, 3) [km]
        delta_d = d2 - d1 # (N_minima, 3) [km]
        delta_t_min = - np.einsum('jk,jk->j', d1, delta_d) / np.linalg.norm(delta_d, axis=1)**2 * dt  # (N_minima,) [s]
        t_min = [dates[minima[i]]+timedelta(seconds=delta_t_min[i]) for i in range(len(minima))] # (N_minima,) [datetime]
        d_min = np.linalg.norm(d1 + delta_d * np.broadcast_to(delta_t_min[...,None], (delta_t_min.shape[0], 3))/dt, axis=1) # (N_minima,) [km]
        
        t_mins.append(np.array(t_min))
        d_mins.append(np.array(d_min))

    return t_mins, d_mins

def sgp4_distance_minimum_solver(rp : np.ndarray, rs : np.ndarray, dates: np.ndarray[datetime], 
                                 primary_satrec, secondary_satrecs, primary_space_object, secondary_space_objects):
    """
        estimates the minima of the distance between a primary objects and a set of secondary objects from the ephimeris positions of the objects.
        rp, rs, dates, satrecs/spaceobjects -> t_mins, d_mins
        rp: (1, N_dates, 3)
        rs: (N_sat, N_dates, 3)
        dates: (N_dates,)

        t_mins: list(ndarray(N_min_i,))
        d_mins: list(ndarray(N_min_i,))
    """
    minimas = _find_distance_minima_in_ephemeris(rp, rs) # N_sat x (N_minima_i,)

    t_mins = []
    d_mins = []
    for secondary_satrec, secondary_space_object, minima in zip(secondary_satrecs, secondary_space_objects, minimas):
        # calculate time bounds for the solver
        minima_jds = [jday(*dates[m].timetuple()[:6]) for m in minima] # found minimas in julian format
        minima_jds = np.array([minima_jds[i][j] for i in range(len(minima_jds)) for j in [0,1]]) # flatten (2*N_minima,)
        lower_bound = [jday(*dates[minima-1][i].timetuple()[:6]) for i in range(len(minima))] # collect lower bounds
        lower_bound = np.array([lower_bound[i][j] for i in range(len(minima)) for j in [0,1]]) #flatten (2*N_minima,)
        upper_bound = [jday(*dates[minima+1][i].timetuple()[:6]) for i in range(len(minima))] # collect upper bounds
        upper_bound = np.array([upper_bound[i][j] for i in range(len(minima)) for j in [0,1]]) #flatten (2*N_minima,)
        bounds = [[lower_bound[i], upper_bound[i]] for i in range(len(minima_jds))]

        # define loss function to be optimized (closure over satrecs)
        loss_function = lambda julian_date_array, secondary_satrec=secondary_satrec, secondary_space_object=secondary_space_object: L_distance_sum(julian_date_array, primary_satrec=primary_satrec, secondary_satrec=secondary_satrec, primary_space_object=primary_space_object, secondary_space_object=secondary_space_object)

        # optimize for minimum loss
        result = scipy.optimize.minimize(loss_function, minima_jds, bounds=bounds)

        # recover dates of distance minimum
        t_min = []
        for jd, jfr in zip(result.x[0::2], result.x[1::2]):
            t_min.append(datetime(*juliandate.to_gregorian(jd+jfr), tzinfo=timezone.utc))        
        d_min = propagate_distance(result.x[0::2], result.x[1::2], primary_satrec, secondary_satrec, primary_space_object, secondary_space_object)

        t_mins.append(np.array(t_min))
        d_mins.append(np.array(d_min))

    return t_mins, d_mins

def screen_for_close_approach(t_mins:list[np.ndarray[datetime]], d_mins:list[np.ndarray[float]], radius : float=100) -> tuple[list[np.ndarray[datetime]], list[np.ndarray[float]]]:
    """
        Filter distance minimum for close approaches. 
        radius:float [km]
        t_mins: list(ndarray[datetime]) (N_sat, N_min)
        d_mins: list(ndarray[float]) (N_sat, N_min)
    """
    # filter for close approaches (<100km) that pierce the screening volume
    d_cas = []
    t_cas = []
    for i in range(len(t_mins)):
        d_ca = np.asarray(d_mins[i])[d_mins[i]<radius]
        t_ca = np.asarray(t_mins[i])[d_mins[i]<radius]
        d_cas.append(d_ca)
        t_cas.append(t_ca)

    return t_cas, d_cas

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

def find_conjunction_track(dates_mins, d_mins, radius, primary_satrec, secondary_satrec, primary_space_object, secondary_space_object):
    """
    dates_mins: (N_ca, ) datetime
    d_mins: (N_ca, ) float
    radius: (1,)

    find entry and exit points of secondary object into screening volume. Then generate trajectory from entry to exit and convert ro rtn
    """

    julian_dates, julian_dates_fractions = ([jday(*dates_mins[j].timetuple()[:6])[0] for j in range(len(dates_mins))], [jday(*dates_mins[j].timetuple()[:6])[1] for j in range(len(dates_mins))]) #(N_ca) jul_date
    rp, vp = propagate_satrec(satrec=primary_satrec,  julian_dates=julian_dates, julian_date_fractions=julian_dates_fractions, space_object = primary_space_object) # (N_ca, 3) # [km] [km/s]
    rs, vs = propagate_satrec(satrec=secondary_satrec,  julian_dates=julian_dates, julian_date_fractions=julian_dates_fractions, space_object = secondary_space_object) # (N_ca, 3) # [km] [km/s]

    r0 = rs - rp # (N_ca, 3) [km]
    v0 = vs - vp # (N_ca, 3) [km/s] relative momentary positions and velocities 

    # best geuss of t_entry and t_exit from linear apporach
    v_dot_r = np.einsum("ni,ni->n", v0, r0) # (N_ca,) [km**2/s]
    v_norm = np.linalg.norm(v0, axis=1) # (N_ca,) [km/s]
    r_norm = np.linalg.norm(r0, axis=1) # (N_ca,) [km]
    t_entry = - v_dot_r/v_norm**2 - 1/v_norm * np.sqrt( v_dot_r**2/v_norm**2 - r_norm**2 + radius**2 ) # (N_ca, )[s]
    t_exit = - v_dot_r/v_norm**2 + 1/v_norm * np.sqrt( v_dot_r**2/v_norm**2 - r_norm**2 + radius**2 ) # (N_ca, )[s]

    date_entry = np.array([dates_mins[i] + timedelta(seconds=t_entry[i]) for i in range(len(t_entry))]) # (N_ca, ) datetime
    date_exit = np.array([dates_mins[i] + timedelta(seconds=t_exit[i]) for i in range(len(t_exit))]) # (N_ca, ) datetime

    julian_date_entry = [jday(*date_entry[m].timetuple()[:6]) for m in range(len(date_entry))] # (N_ca, 2)
    julian_date_entry = np.array([julian_date_entry[i][j] for i in range(len(julian_date_entry)) for j in [0,1]]) # flatten (2*N_ca,)
    julian_date_exit = [jday(*date_exit[m].timetuple()[:6]) for m in range(len(date_exit))] # (N_ca, 2)
    julian_date_exit = np.array([julian_date_exit[i][j] for i in range(len(julian_date_exit)) for j in [0,1]]) # flatten (2*N_ca,)

    lower_bound_entry = [jday(*(date_entry[i]-timedelta(seconds=2)).timetuple()[:6]) for i in range(len(date_entry))] # collect lower bounds
    lower_bound_entry = np.array([lower_bound_entry[i][j] for i in range(len(lower_bound_entry)) for j in [0,1]]) #flatten (2*N_ca,)
    upper_bound_entry = [jday(*dates_mins[i].timetuple()[:6]) for i in range(len(dates_mins))] # collect upper bounds
    upper_bound_entry = np.array([upper_bound_entry[i][j] for i in range(len(upper_bound_entry)) for j in [0,1]]) #flatten (2*N_ca,)
    bounds_entry = [[lower_bound_entry[i], upper_bound_entry[i]] for i in range(len(julian_date_entry))]

    upper_bound_exit = [jday(*(date_exit[i]+timedelta(seconds=2)).timetuple()[:6]) for i in range(len(date_exit))] # collect upper bounds
    upper_bound_exit = np.array([upper_bound_exit[i][j] for i in range(len(upper_bound_exit)) for j in [0,1]]) #flatten (2*N_ca,)
    lower_bound_exit = [jday(*dates_mins[i].timetuple()[:6]) for i in range(len(dates_mins))] # collect lower bounds
    lower_bound_exit = np.array([lower_bound_exit[i][j] for i in range(len(lower_bound_exit)) for j in [0,1]]) #flatten (2*N_ca,)
    bounds_exit = [[lower_bound_exit[i], upper_bound_exit[i]] for i in range(len(julian_date_exit))]

    # solve sgp4 for entry and exit points with t_entry and t_exit as initial guess for solver
    loss_function = lambda julian_date_array, secondary_satrec=secondary_satrec, secondary_space_object=secondary_space_object: L_distance_sum(julian_date_array, primary_satrec=primary_satrec, secondary_satrec=secondary_satrec, primary_space_object=primary_space_object, secondary_space_object=secondary_space_object, target_distance=radius)
    result_entry = scipy.optimize.minimize(loss_function, julian_date_entry, bounds=bounds_entry)
    result_exit = scipy.optimize.minimize(loss_function, julian_date_exit, bounds=bounds_exit)


    print(result_entry)
    print(result_exit)
    t_entry = []
    for jd, jfr in zip(result_entry.x[0::2], result_entry.x[1::2]):
        t_entry.append(datetime(*juliandate.to_gregorian(jd+jfr), tzinfo=timezone.utc))        
    d_entry = propagate_distance(result_entry.x[0::2], result_entry.x[1::2], primary_satrec, secondary_satrec, primary_space_object, secondary_space_object)
    t_exit = []
    for jd, jfr in zip(result_exit.x[0::2], result_exit.x[1::2]):
        t_exit.append(datetime(*juliandate.to_gregorian(jd+jfr), tzinfo=timezone.utc))        
    d_exit = propagate_distance(result_exit.x[0::2], result_exit.x[1::2], primary_satrec, secondary_satrec, primary_space_object, secondary_space_object)

    print(d_entry)
    print(d_exit)

    # generate trajectory between entry and exit
    conjunction_tracks = []
    for start, end in zip(t_entry, t_exit):
        r_track_p, v_track_p, track_dates_p = generate_ephemeris(satrecs=primary_satrec, space_object=primary_space_object, start_date=start, end_date=end, N_dates = 50)
        r_track_s, v_track_s, track_dates_s = generate_ephemeris(satrecs=secondary_satrec, space_object=secondary_space_object, start_date=start, end_date=end, N_dates = 50)
        rs_rtn, vs_rtn = TEME_2_RTN(r_track_p, v_track_p, r_track_s, v_track_s)
        print(np.linalg.norm(rs_rtn, axis=1))
        conjunction_tracks.append(rs_rtn)

    return t_entry, t_exit, conjunction_tracks
    
def conjunction_screening(primary_space_object : SpaceObject, 
                          secondary_space_objects : list[SpaceObject], 
                          primary_satrec : Satrec, 
                          secondary_satrecs : list[Satrec],
                          start_time: datetime,
                          screen_interval : float = 2,
                          dt : float = 2) -> CDMT:
    """
    screen for conjunctions for a primary satellite of interest over a time window.

    screen_interval : scrren interval [hours]
    dt : scan time resolution [s]
    """

    # propagate satrecs over screen interval (ephimeris)
    end_time = start_time+timedelta(hours=screen_interval)
    N = int((end_time - start_time) / timedelta(seconds=dt))
    rp, _, dp = generate_ephemeris(primary_satrec, primary_space_object, start_time, end_time, N) # (N_t [km], 3), (N_t [km/s], 3), (N_t [datetime])
    rs, _, _ = generate_ephemeris_array(SatrecArray(secondary_satrecs), secondary_space_objects, start_time, end_time, N) # (N_t [km], 3), (N_t [km/s], 3), (N_t [datetime])
    
    #linear distance minimum estimate
    #t_mins, d_mins = linear_distance_minimum_estimate(rp, rs, dp) # (N_sat,N_min), (N_sat,N_min)

    # use sgp4  and optimize for minimas in the distances of the screen interval
    t_mins_sgp4, d_mins_sgp4 = sgp4_distance_minimum_solver(rp, rs, dp,
                                                            primary_satrec, secondary_satrecs, primary_space_object, secondary_space_objects) # (N_sat,N_min), (N_sat,N_min)

    # screen for close conjunctions
    conjunction_alert_distance_threshold = 100 # [km]
    TCAs, DCAs = screen_for_close_approach(t_mins_sgp4, d_mins_sgp4, radius=conjunction_alert_distance_threshold) # (N_sat, N_ca)

    cdms = []
    cdm_tracks = []
    # for each secondary satellite
    for i in range(len(secondary_space_objects)):
        # at each close approach calculate: relative_speed, relative_rtn_pos, relative_rtn_velocities
        dates = TCAs[i]
        distances = DCAs[i]
        julian_dates, julian_dates_fractions = ([jday(*dates[j].timetuple()[:6])[0] for j in range(len(dates))], [jday(*dates[j].timetuple()[:6])[1] for j in range(len(dates))]) 
        rp, vp = propagate_satrec(satrec=primary_satrec,  julian_dates=julian_dates, julian_date_fractions=julian_dates_fractions, space_object = primary_space_object) # (N_ca, 3)
        rs, vs = propagate_satrec(satrec=secondary_satrecs[i],  julian_dates=julian_dates, julian_date_fractions=julian_dates_fractions, space_object = secondary_space_objects[i]) # (N_ca, 3)

        relative_speed = np.linalg.norm(vs-vp, axis=1) # (N_ca,)

        rs_rtn, vs_rtn = TEME_2_RTN(rp, vp, rs, vs) # (N_ca, 3)

        dates_entry, dates_exit, conjunction_tracks = find_conjunction_track(dates, distances, conjunction_alert_distance_threshold,
                                                                            primary_satrec, secondary_satrecs[i], primary_space_object, secondary_space_objects[i])
        
        

        # generate CDMs
        for d, t, delta_v, rtn_pos, rtn_vel, entry_time, exit_time, track in zip(distances, dates, relative_speed, rs_rtn, vs_rtn, dates_entry, dates_exit, conjunction_tracks):
            # TODO 
            # # find collision probability
            # clean up code
            # visualization
            cdm = CDMT(CCSDS_CDM_VERS="1.0",
                    CREATION_DATE=datetime.now(tz=UTC).strftime("%m-%d-%YT%H:%M:%S.%f".rstrip('0')),
                    ORIGINATOR=None,
                    MESSAGE_FOR=primary_space_object.name,
                    MESSAGE_ID=None,
                    TCA=t.strftime("%m-%d-%YT%H:%M:%S.%f".rstrip('0')),
                    MISS_DISTANCE=float(d*1000),
                    RELATIVE_SPEED=float(delta_v*1000),
                    START_SCREEN_PERIOD=start_time.strftime("%m-%d-%YT%H:%M:%S.%f".rstrip('0')),
                    STOP_SCREEN_PERIOD=end_time.strftime("%m-%d-%YT%H:%M:%S.%f".rstrip('0')),
                    SCREEN_VOLUME_SHAPE="ELLIPSOID",
                    COLLISION_PROBABILITY=None,
                    OBJECT1=primary_space_object.name,
                    OBJECT2=secondary_space_objects[i].name,
                    RELATIVE_POSITION_R=float(rtn_pos[0]*1000),
                    RELATIVE_POSITION_T=float(rtn_pos[1]*1000),
                    RELATIVE_POSITION_N=float(rtn_pos[2]*1000),
                    RELATIVE_VELOCITY_R=float(rtn_vel[0]*1000),
                    RELATIVE_VELOCITY_T=float(rtn_vel[1]*1000),
                    RELATIVE_VELOCITY_N=float(rtn_vel[2]*1000),
                    SCREEN_VOLUME_X=float(conjunction_alert_distance_threshold*1000),
                    SCREEN_VOLUME_Y=float(conjunction_alert_distance_threshold*1000),
                    SCREEN_VOLUME_Z=float(conjunction_alert_distance_threshold*1000),
                    SCREEN_VOLUME_FRAME="RTN",
                    SCREEN_ENTRY_TIME = entry_time.strftime("%m-%d-%YT%H:%M:%S.%f".rstrip('0')),
                    SCREEN_EXIT_TIME = exit_time.strftime("%m-%d-%YT%H:%M:%S.%f".rstrip('0'))
                    )
            cdms.append(cdm)
            cdm_tracks.append(track)

    return cdms, cdm_tracks

if __name__ == "__main__":

    # query primary
    primary_space_objects, primary_satrecs = query("ISS (ZARYA)", query_type="NAME") 
    primary_satrec = primary_satrecs[0]
    primary_space_object = primary_space_objects[0]
    # query secondary
    secondary_space_objects,secondary_satrecs = query("Tianhe", query_type="NAME")
    # remove primary from secondary if needed
    for so in secondary_space_objects:
        if so.norad_id == primary_space_object.norad_id:
            secondary_space_objects.remove(so)


    cdms, cdm_tracks = conjunction_screening(primary_space_object, secondary_space_objects, primary_satrec, secondary_satrecs, start_time=datetime.now(tz=timezone.utc), screen_interval=4*24)

    from pprint import pprint
    for cdm in cdms:
        pprint(vars(cdm), sort_dicts=False)

    for cdm, track in zip(cdms, cdm_tracks):

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        radius = 100

        # Primary satellite name from CDM
        primary_name = cdm.OBJECT1  # Adjust if CDM is an object rather than a dict

        # Plot conjunction trajectory
        ax.plot(
            track[:, 0],
            track[:, 1],
            track[:, 2],
            color='red',
            linewidth=2,
            label='Secondary trajectory'
        )

        # Create transparent sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)

        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones_like(u), np.cos(v))

        ax.plot_surface(
            x, y, z,
            color='lightblue',
            alpha=0.12,
            edgecolor='none'
        )

        # Primary satellite at origin
        ax.scatter(
            0, 0, 0,
            color='blue',
            marker='x',
            s=120,
            linewidths=3,
            label=f'Primary: {primary_name}'
        )

        ax.text(
            0, 0, 0,
            f'  {primary_name}',
            color='blue',
            fontsize=10
        )

        # Entry and exit points
        ax.scatter(*track[0], color='green', s=60, label='Entry')
        ax.scatter(*track[-1], color='orange', s=60, label='Exit')

        # Find closest approach
        distances = np.linalg.norm(track, axis=1)
        idx_min = np.argmin(distances)

        min_point = track[idx_min]
        min_distance = distances[idx_min]

        # Mark closest approach
        ax.scatter(
            *min_point,
            color='black',
            s=80,
            label='Closest approach'
        )

        # Line from primary to closest approach
        ax.plot(
            [0, min_point[0]],
            [0, min_point[1]],
            [0, min_point[2]],
            color='black',
            linestyle='--',
            linewidth=1.5
        )

        # Annotate minimum distance at midpoint
        midpoint = min_point / 2

        ax.text(
            midpoint[0],
            midpoint[1],
            midpoint[2],
            f'  {cdm.MISS_DISTANCE:.3f} km',
            color='black',
            fontsize=10
        )

        # Axes
        ax.set_xlabel('Radial R [km]')
        ax.set_ylabel('Transverse T [km]')
        ax.set_zlabel('Normal N [km]')

        ax.set_title(f'Conjunction trajectory — {primary_name}')

        ax.set_box_aspect((1, 1, 1))

        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_zlim(-radius, radius)

        ax.legend()
        plt.tight_layout()
        plt.show()
