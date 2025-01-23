
"""
Satellite Deorbiting Mission Planning Simulation

Author: Rachit Bhatia
Version: 3.1
Date: 2025-01-22

Objective:
----------
A reusable tool for end-of-life mission planning and enabling controlled deorbit of
LEO satellites while maximizing fuel consumption to ensure safe and secured end-of-life disposal.
This tool uses simplified dynamics with advanced orbital mechanics and numerical integration techniques.
The script calculates optimal burn sequences to lower satellite orbits and manage propellant
consumption during atmospheric reentry.

Inputs:
-------
- Satellite orbital parameters:
  * Semi-major axis
  * Eccentricity
  * Inclination
  * Right Ascension of Ascending Node (RAAN)
  * Argument of Perigee
  * True Anomaly
- Satellite physical characteristics:
  * Drag surface area
  * Dry mass
  * Initial propellant mass
  * Thrust capability
  * Specific impulse
  * Drag coefficient
- Mission parameters:
  * Initial mission epoch
  * Number of satellites to simulate

Outputs:
--------
- Optimal burn start times for each satellite
- Remaining propellant after deorbiting sequence
- Visualization of:
  * Satellite position vs time
  * Altitude vs time
  * Propellant consumption vs time

Dependencies:
-------------
- NumPy
- SciPy
- Poliastro
- Astropy
- Matplotlib
- Multiprocessing

Methodology:
------------
Uses Gauss planetary equations and Runge-Kutta 45 numerical integration to model
satellite orbital dynamics, atmospheric drag, and thrust-assisted deorbiting. This version uses
orbital_elements_to_inertial method to remove some approximations for calculating position
and velocity magnitude in the Inertial frame. This script also takes into account the rotation
of the Earth's atmosphere while calculating acceleration due to drag.
"""

import numpy as np
from datetime import datetime, timedelta, timezone
from scipy.integrate import solve_ivp
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy import units as u
import matplotlib.pyplot as plt

import multiprocessing
from multiprocessing import Pool

# Constants
mu = Earth.k.to(u.m ** 3 / u.s ** 2).value  # Earth's gravitational parameter, m^3/s^2
Re = 6.3781363e6  # Earth's radius, m
g0 = 9.81  # Earth's gravitational acceleration, m/s^2
we = 7.2921159e-5  # Earth's rotation rate (rad/s)

rho60 = 3.206e-04  # Atmospheric density at 60 km altitude, kg/m^3
H = 7.714e3  # Scale height, m
RefAlt = 60e3  # Reference Altitude, m

rtol_val = 1e-09
atol_val = 1e-11


def convert_and_range_angle(degrees, min_angle_rad, max_angle_rad):
    '''Convert degrees to radians and ensure the angle is within a specified range.

        Args:
            degrees (float): Angle in degrees
            min_angle_rad (float): Minimum allowed angle in radians
            max_angle_rad (float): Maximum allowed angle in radians

        Returns:
            float: Angle in radians, wraps the angle
        '''
    # Convert degrees to radians using numpy
    angle_rad = np.deg2rad(degrees)
    # Ensure the angle stays within the specified range
    angle_rad = np.arctan2(np.sin(angle_rad), np.cos(angle_rad))
    return angle_rad

def atmospheric_density(altitude):
    '''Calculate atmospheric density at a given altitude using an exponential model.

        Args:
            altitude (float): Altitude in meters

        Returns:
            float: Atmospheric density in kg/m^3
        '''
    if altitude > 800e3:  # Beyond 800 km altitude, assume near-zero atmospheric density
        return 0
    return rho60 * np.exp(-(altitude - RefAlt) / H)

# Function to compute drag acceleration
def drag_acceleration(r_vec, v_vec, rho, Cd, S, m0, mp):
    '''Compute drag acceleration based on satellite parameters and atmospheric conditions.

        Args:
            r (numpy.ndarray): Position vector in meters
            v (numpy.ndarray): Velocity vector in meters per seconds
            rho (float): Atmospheric density in kg/m^3
            Cd (float): Drag coefficient
            S (float): Cross-sectional area in meter squared
            m0 (float): Dry mass of satellite in kg
            mp (float): Propellant mass in kg

        Returns:
            numpy.ndarray: Drag acceleration magnitude
        '''
    # Magnitude of position & velocity
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    v_rel = v - np.cross(np.array([0, 0, we]), r_vec)
    drag_acc = -0.5 * rho * (Cd * S / (m0 + mp)) * np.linalg.norm(v_rel) * v_rel
    return drag_acc

def orbital_elements_to_inertial(a, e, i, RAAN, omega, nu):
    '''Convert classical orbital elements to inertial position and velocity using poliastro.'''
    # Convert orbital elements to Orbit object
    orbit = Orbit.from_classical(Earth, a * u.m, e * u.one, np.rad2deg(i) * u.deg, np.rad2deg(RAAN) * u.deg,
                                 np.rad2deg(omega) * u.deg, np.rad2deg(nu) * u.deg)

    # Get position and velocity in ECI frame (meters, meters/second)
    r_vec, v_vec = orbit.r.value * 1000, orbit.v.value * 1000
    return r_vec, v_vec

def compute_acceleration(r_vec, v_vec, m0, m_propellant, S, Cd, F, burn_flag):
    '''Compute acceleration vector in Radial-Tangential-Normal (RTN) frame.

    Args:
        r (numpy.ndarray): Position vector in meters
        v (numpy.ndarray): Velocity vector in meters per seconds
        m0 (float): Dry mass of satellite in kg
        m_propellant (float): Propellant mass in kg
        S (float): Cross-sectional area in meters squared
        Cd (float): Drag coefficient
        F (float): Thrust force in Newton
        burn_flag (int): Flag indicating whether thrust is active, 1 means satellite is thrusting

    Returns:
        numpy.ndarray: Acceleration vector in RTN frame
    '''
    # Magnitude of position & velocity
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    altitude = r - Re  # current altitude, meters
    rho = atmospheric_density(np.linalg.norm(altitude))  # Ensure altitude is in meters for density calculation

    # Drag force
    acc_drag = drag_acceleration(r_vec, v_vec, rho, Cd, S, m0, m_propellant)  # Drag force

    if burn_flag == 1:
        acc_thrust = - F / (m0 + m_propellant) # acceleration due to thrust
    else:
        acc_thrust = 0

    acc_gravity = - ( mu / r**3 ) * r_vec #-9.81  # m per seconds squared
    acc_rtn = np.array([-np.linalg.norm(acc_gravity), (-np.linalg.norm(acc_drag) + acc_thrust), 0])  # RTN acceleration

    return acc_rtn


def gauss_planetary_equations(t, y, S, m0, F, Isp, Cd, Initial_epoch, burn_start_time_epoch, burn_start_time_delta,
                              burn_flag):
    '''Compute the derivatives of orbital elements using Gauss planetary equations.

        Args:
            t (float): Time in seconds
            y (list): Current state variables [a, e, i, RAAN, omega, nu, m_propellant, altitude]
            S (float): Cross-sectional area in meters squared
            m0 (float): Dry mass of satellite in kg
            F (float): Thrust force in Newton
            Isp (float): Specific impulse in seconds
            Cd (float): Drag coefficient
            Initial_epoch (datetime): Mission start time
            burn_start_time_epoch (datetime): Burn start time
            burn_start_time_delta (float): Time since burn start in seconds
            burn_flag (int): Flag indicating thrust status, 1 means satellite is thrusting

        Returns:
            list: Derivatives of state variables with respect to time
        '''
    a, e, i, RAAN, omega, nu, m_propellant, current_altitude = y
    r_vec, v_vec = orbital_elements_to_inertial(a, e, i, RAAN, omega, nu)
    # Magnitude of position & velocity
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    current_epoch = Initial_epoch + timedelta(seconds=t)
    current_epoch = current_epoch.strftime("%Y-%m-%dT%H:%M:%SZ")

    burn_start_time_epoch = burn_start_time_epoch + timedelta(seconds=burn_start_time_delta)
    burn_start_time_epoch = burn_start_time_epoch.strftime("%Y-%m-%dT%H:%M:%SZ")

    if (current_epoch >= burn_start_time_epoch):
        burn_flag = 1

    i = convert_and_range_angle(np.rad2deg(i), 0, np.pi)  # Inclination, degrees
    RAAN = convert_and_range_angle(np.rad2deg(RAAN), -np.pi, np.pi)  # RAAN, degrees
    omega = convert_and_range_angle(np.rad2deg(omega), -np.pi, np.pi)  # Argument of perigee, degrees
    nu = convert_and_range_angle(np.rad2deg(nu), -np.pi, np.pi)  # True anomaly, degrees

    p = a * (1 - e ** 2)
    # r = p / (1 + e * np.cos(nu))
    specific_ang_momentum = check_and_handle_imaginary_sqrt(mu * p)
    # v = check_and_handle_imaginary_sqrt(2 * ((-mu / (2 * a)) + (mu / r)))
    true_longitude = convert_and_range_angle(np.rad2deg(omega + nu), -np.pi, np.pi)
    c = Isp * g0

    altitude = (r - Re) / 1000

    # Compute the RTN acceleration (drag)
    acc_rtn = compute_acceleration(r_vec, v_vec, m0, m_propellant, S, Cd, F, burn_flag)

    # Calculate derivatives using the provided equations
    if (specific_ang_momentum > 0) and (r > 0) and (np.sin(i)):
        da_dt = (2 * a ** 2 / specific_ang_momentum) * (e * np.sin(nu) * acc_rtn[0] + p * acc_rtn[1] / r)
        de_dt = (p / specific_ang_momentum) * (
                    np.sin(nu) * acc_rtn[0] + ((np.cos(nu) + (e + np.cos(nu)) / (1 + e * np.cos(nu))) * acc_rtn[1]))
        di_dt = (r / specific_ang_momentum) * acc_rtn[2] * np.cos(true_longitude)
        dRAAN_dt = (r / specific_ang_momentum) * acc_rtn[2] * np.sin(true_longitude) / np.sin(i)
        domega_dt = (1 / (specific_ang_momentum * e)) * (
                    -p * np.cos(nu) * acc_rtn[0] + (p + r) * np.sin(nu) * acc_rtn[1]) - dRAAN_dt * np.cos(i)
        dnu_dt = (specific_ang_momentum / r ** 2) - domega_dt - dRAAN_dt * np.cos(i)
    else:
        da_dt = 0
        de_dt = 0
        di_dt = 0
        dRAAN_dt = 0
        domega_dt = 0
        dnu_dt = 0

    # Calculate dm_dt
    if (burn_flag == 1) and (c > 0):
        dm_propellant_dt = - F / c  # Assuming constant c
    else:
        dm_propellant_dt = 0

    dalt_dt = (altitude - current_altitude)

    # print(f"t: {t}")
    # print(f"Altitude: {altitude} km")
    # print(f"m_propellant: {m_propellant} kg")

    return [da_dt, de_dt, di_dt, dRAAN_dt, domega_dt, dnu_dt, dm_propellant_dt, dalt_dt]


# Terminal event function to stop integration when altitude is below 100 km
def terminate_altitude_event(t, y, *args):
    '''Event function to terminate the simulation when the satellite reaches a target altitude.

        Args:
            t (float): Current time in seconds
            y (array): Current state vector
            *args: Additional arguments

        Returns:
            float: Event value for altitude termination condition
        '''
    try:
        target_altitude = 98.0  # in km
        altitude = y[7]
        event_value = np.floor(altitude - target_altitude)
        # print(f"Time: {t}, Altitude: {altitude}, Event Value: {event_value}")
        return event_value
    except Exception as e:
        print(f"Error in terminate_event: {e}, t={t}, y={y}, args={args}")
        return 0  # Safe fallback to avoid NoneType issues

# Terminal event function to stop integration when propellant is empty
def terminate_propellant_event(t, y, *args):
    '''Event function to terminate the simulation when propellant is almost depleted.

        Args:
            t (float): Current time in seconds
            y (array): Current state vector
            *args: Additional arguments

        Returns:
            float: Event value for propellant termination condition
        '''
    try:
        target_propellant_level = 0.0001  # in kg
        current_propellant_level = y[6]
        event_value = current_propellant_level - target_propellant_level  # Replace with your condition
        # print(f"Time: {t}, Propellant Level: {current_propellant_level}, Event Value: {event_value}")
        return event_value
    except Exception as e:
        print(f"Error in terminate_event: {e}, t={t}, y={y}, args={args}")
        return 0  # Safe fallback to avoid NoneType issues

# Terminal event properties
terminate_altitude_event.terminal = True  # Stop integration when the event is triggered
terminate_altitude_event.direction = -1  # Trigger only when crossing zero in the negative direction

terminate_propellant_event.terminal = True  # Stop integration when the event is triggered
terminate_propellant_event.direction = -1  # Trigger only when crossing zero in the negative direction


def solve_ode(y0, t_span, t_eval, burn_start_time_delta, S, m0, mp, F, Isp, Cd, Initial_epoch, burn_start_time_epoch,
              burn_flag, rtol_val, atol_val):
    '''Propagate satellite orbit using Gauss planetary equations and Runge-Kutta 45 numerical integrator.

        Args:
            y0 (list): Initial state vector
            t_span (tuple): Time span for simulation in seconds
            t_eval (array): Time points for evaluation in seconds
            burn_start_time_delta (float): Time delta for burn start in seconds
            S (float): Drag surface area in meters squared
            m0 (float): Dry mass of satellite in kg
            mp (float): Initial propellant mass in kg
            F (float): Thrust in Newton
            Isp (float): Specific impulse in seconds
            Cd (float): Drag coefficient
            Initial_epoch (datetime): Initial simulation epoch in UTC
            burn_start_time_epoch (datetime): Burn start time epoch in UTC
            burn_flag (int): Flag indicating thrust state, 1 means satellite is thrusting
            rtol_val (float): Relative tolerance for integration
            atol_val (float): Absolute tolerance for integration

        Returns:
            Propagated state and corresponding time, solution of Gauss Planetary equations using solve_ivp
        '''

    terminal_events = [terminate_altitude_event, terminate_propellant_event]

    sol = solve_ivp(gauss_planetary_equations, t_span, y0, t_eval=t_eval,
                    args=(S, m0, F, Isp, Cd, Initial_epoch, burn_start_time_epoch, burn_start_time_delta, burn_flag),
                    method="RK45", rtol=rtol_val, atol=atol_val, events=terminal_events)

    return sol


def optimize_burn_sequence(sol, burn_start_time_delta, y0, t_span, t_eval, S, m0, mp, F, Isp, Cd, Initial_epoch,
                           burn_start_time_epoch,
                           burn_flag, delta_optimize_seconds, iteration_flag, convergence_flag, Num_iter,
                           Max_iter_allowed):
    '''Optimize the burn sequence to achieve deorbiting with maximum propellant usage.

        This function iteratively adjusts burn start time to:
        1. Reach 100 km altitude
        2. Maximize propellant consumption

        Args:
            sol (ODE solution): Initial ODE solution
            burn_start_time_delta (float): Initial burn start time delta in seconds
            y0 (list): Initial state vector
            t_span (tuple): Time span for simulation in seconds
            t_eval (array): Time evaluation points in seconds
            S, m0, mp, F, Isp, Cd (float): Satellite physical parameters
            Initial_epoch, burn_start_time_epoch (datetime): Epoch times
            burn_flag (int): Flag indicating thrust state, 1 means satellite is thrusting
            delta_optimize_seconds (float): Time increment for optimization in seconds
            iteration_flag (int): Flag to track iteration state
            convergence_flag (int): Flag to track convergence
            Num_iter (int): Current iteration number
            Max_iter_allowed (int): Maximum allowed iterations

        Returns:
            tuple: Optimized simulation times, states, burn time, remaining propellant, and convergence status
        '''
    times = sol.t
    states = sol.y

    print(f"Iteration Number:  {Num_iter + 1}")

    burn_start_time = None
    for idx, (altitude, propellant) in enumerate(
            zip(states[7], states[6])):
        if altitude < 100:  # Reached deorbit altitude
            if burn_start_time is None:
                burn_start_time = burn_start_time_delta / 60  # Time in minutes
                # print(f"burn_start_time: {burn_start_time}")
                convergence_flag = 1
                print(
                    f"Iteration Number:  {Num_iter + 1}. \n Reached 100km altitude and have remaining propellant. \n Last altitude: {altitude} km, \n Optimal burn start time from the initial epoch: {burn_start_time} minutes, \n Remaining propellant: {states[6][-1]} kg")
                break
        if propellant < 1e-3:  # If propellant remaining is less than 1 gram
            if burn_start_time is None:
                while ((convergence_flag == 0) and (Num_iter < Max_iter_allowed)):
                    burn_start_time_delta = burn_start_time_delta + delta_optimize_seconds
                    new_sol = []
                    new_sol = solve_ode(y0, t_span, t_eval, burn_start_time_delta, S, m0, mp, F, Isp, Cd, Initial_epoch,
                                        burn_start_time_epoch,
                                        burn_flag, rtol_val, atol_val)
                    times = []
                    states = []
                    times = new_sol.t
                    states = new_sol.y
                    iteration_flag = 1
                    Num_iter = Num_iter + 1
                    [times, states, burn_start_time, remaining_propellant, convergence_flag] = optimize_burn_sequence(
                        new_sol, burn_start_time_delta, y0, t_span, t_eval, S, m0, mp, F, Isp, Cd, Initial_epoch,
                        burn_start_time_epoch,
                        burn_flag, delta_optimize_seconds, iteration_flag, convergence_flag, Num_iter, Max_iter_allowed)
                break

    if burn_start_time is None and iteration_flag == 0:
        while ((convergence_flag == 0) and (Num_iter < Max_iter_allowed)):
            burn_start_time_delta = burn_start_time_delta + delta_optimize_seconds
            new_sol = []
            new_sol = solve_ode(y0, t_span, t_eval, burn_start_time_delta, S, m0, mp, F, Isp, Cd, Initial_epoch,
                                burn_start_time_epoch,
                                burn_flag, rtol_val, atol_val)
            times = new_sol.t
            states = new_sol.y
            iteration_flag = 1
            Num_iter = Num_iter + 1
            [times, states, burn_start_time, remaining_propellant, convergence_flag] = optimize_burn_sequence(new_sol,
                                                                                                              burn_start_time_delta,
                                                                                                              y0,
                                                                                                              t_span,
                                                                                                              t_eval, S,
                                                                                                              m0, mp, F,
                                                                                                              Isp, Cd,
                                                                                                              Initial_epoch,
                                                                                                              burn_start_time_epoch,
                                                                                                              burn_flag,
                                                                                                              delta_optimize_seconds,
                                                                                                              iteration_flag,
                                                                                                              convergence_flag,
                                                                                                              Num_iter,
                                                                                                              Max_iter_allowed)

    if (convergence_flag == 1) and (burn_start_time is not None):
        remaining_propellant = states[6][-1]  # Remaining propellant after burn
    else:
        remaining_propellant = mp  # Remaining propellant after burn

    if (Num_iter >= Max_iter_allowed):

        if (convergence_flag == 0):
            remaining_propellant = mp  # Remaining propellant after burn
            print("Max iterations have been exceeded and no convergence!!")
            print(
                f"Iteration Number:  {Num_iter + 1}. \n Did not reach 100km altitude and have remaining propellant. \n Last altitude: {altitude} km, \n Optimal burn time from the initial epoch: {0} minutes, \n Remaining propellant: {mp} kg")
            convergence_flag = 1

    return times, states, burn_start_time, remaining_propellant, convergence_flag


def run_simulation(satellite_data):
    '''Run a complete deorbiting simulation for a single satellite. Processes satellite parameters, performs orbital propagation,
        optimizes burn sequence, and generates results.
        '''
    # unpack the satellite data
    a = satellite_data.get("a")
    e = satellite_data.get("e")
    i = satellite_data.get("i")
    RAAN = satellite_data.get("RAAN")
    omega = satellite_data.get("omega")
    nu = satellite_data.get("nu")
    S = satellite_data.get("S")
    m0 = satellite_data.get("m0")
    mp = satellite_data.get("mp")
    F = satellite_data.get("F")
    Isp = satellite_data.get("Isp")
    Cd = satellite_data.get("Cd")
    Initial_epoch = satellite_data.get("Initial_epoch")
    burn_start_time_epoch = Initial_epoch # Assuming that the default burn start time epoch is at the initial epoch
    burn_flag = satellite_data.get("burn_flag")
    num_sat = satellite_data.get("num_sat")  # Satellite number

    # Generate simulation
    print("\n\n")
    print(f"Running simulation for satellite {num_sat + 1}:")
    print("\n\n")

    altitude0 = (a * (1 - e ** 2) - Re) / 1000
    y0 = [a, e, i, RAAN, omega, nu, mp, altitude0]
    t_span = (0, 30 * 24 * 3600)  # Simulate for 30 days
    t_eval = np.arange(0, t_span[1], 1)  # Evaluate at every second
    burn_start_time_delta = 0

    sol0 = solve_ode(y0, t_span, t_eval, burn_start_time_delta, S, m0, mp, F, Isp, Cd, Initial_epoch,
                     burn_start_time_epoch,
                     burn_flag, rtol_val, atol_val)

    iteration_flag = 0
    convergence_flag = 0
    Num_iter = 0
    Max_iter_allowed = 10
    delta_optimize_seconds = 30  # in seconds
    [all_times, all_states, burn_time, remaining_propellant, convergence_flag] = optimize_burn_sequence(sol0,
                                                                                                        burn_start_time_delta,
                                                                                                        y0, t_span,
                                                                                                        t_eval, S, m0,
                                                                                                        mp, F, Isp, Cd,
                                                                                                        Initial_epoch,
                                                                                                        burn_start_time_epoch,
                                                                                                        burn_flag,
                                                                                                        delta_optimize_seconds,
                                                                                                        iteration_flag,
                                                                                                        convergence_flag,
                                                                                                        Num_iter,
                                                                                                        Max_iter_allowed)

    if burn_time is not None:
        print(f"Optimal burn start time: {burn_time} minutes")
        print(f"Remaining propellant: {remaining_propellant} kg")
    else:
        convergence_flag = 0
        print(f"Remaining propellant: {remaining_propellant} kg")
        print("No convergence, please try again with increases value of delta_optimize_seconds.")

    # Plot results
    plot_results(num_sat, all_times, all_states)
    results_main = [all_times, all_states, burn_time, remaining_propellant, convergence_flag]

    return results_main


def plot_results(num_sat, all_times, all_states):
    '''Plot position vs time, altitude vs time, and propellant consumption vs time.'''
    a = all_states[0, :]
    e = all_states[1, :]
    nu = all_states[5, :]
    r = a * (1 - e ** 2) / (1 + e * np.cos(nu))  # Radius
    altitude = all_states[7, :]
    propellant_levels = all_states[6, :]

    # Find the first index where the altitude value becomes less than 100 km
    threshold_index_altitude_levels = np.argmax((altitude - 100) < 0.00001)
    threshold_value_altitude_levels = altitude[threshold_index_altitude_levels]

    # Find the first index where the propellant levels value becomes less than 1 gram
    threshold_index_propellant_levels = np.argmax((propellant_levels - 0.001000) < 0.00001)
    threshold_value_propellant_levels = propellant_levels[threshold_index_propellant_levels]

    # Find maximum plotting idx
    if (threshold_index_altitude_levels == threshold_index_propellant_levels):
        max_plot_idx = threshold_index_altitude_levels

    elif (threshold_index_altitude_levels > threshold_index_propellant_levels):
        max_plot_idx = threshold_index_altitude_levels

    else:
        max_plot_idx = threshold_index_propellant_levels

    plot_altitude = altitude[0:max_plot_idx]
    plot_propellant_levels = propellant_levels[0:max_plot_idx]
    plot_times = all_times[0:max_plot_idx]
    plot_r = r[0:max_plot_idx]

    # Plot position vs time
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(plot_times / 60, plot_r / 1e3)  # Time in minutes, radius in km
    plt.xlabel("Time (min)")
    plt.ylabel("Radius (km)")
    plt.title(f"Position vs. Time, for sat. no.: {num_sat + 1}")
    plt.grid()

    # Plot altitude vs time
    plt.subplot(1, 3, 2)
    plt.plot(plot_times / 60, plot_altitude)  # Time in minutes, altitude in km
    plt.xlabel("Time (min)")
    plt.ylabel("Altitude (km)")
    plt.title(f"Altitude vs. Time, for sat. no.: {num_sat + 1}")
    plt.grid()

    # Plot remaining propellant vs time
    plt.subplot(1, 3, 3)
    plt.plot(plot_times / 60, plot_propellant_levels)  # Time in minutes, altitude in km
    plt.xlabel("Time (min)")
    plt.ylabel("Remaining Propellant (kg)")
    plt.title(f"Rem. Propellant vs. Time, for sat. no.: {num_sat + 1}")
    plt.grid()

    plt.tight_layout()
    plt.show()


def check_and_handle_imaginary_sqrt(x):
    try:
        if x < 0:
            # Handle imaginary result here, e.g., return 0 or raise an exception
            return 0
        else:
            result = np.sqrt(x)
            return result
    except ValueError as e:
        print("Error:", e)
        # Handle the exception here, e.g., return a default value
        return None


def main():
    '''
       Main program to simulate controlled deorbiting for multiple satellites.
       Supports sequential and parallel processing.
       '''
    # User inputs
    n = int(input("Enter the number of satellites for which deorbiting mission needs to be planned: "))
    use_parallel = input("Use parallel processing? (yes/no): ").strip().lower() == "yes"

    all_main_data = []
    all_results_main = []

    for num_sat in range(n):
        main_data = {}
        print(f"Enter details for satellite {num_sat + 1}:")

        main_data = {
            "a": float(input("Enter semi-major axis (m): ")),  # Semi-major axis in meters
            "e": float(input("Enter eccentricity: ")),  # Eccentricity
            "i": convert_and_range_angle(float(input("Enter inclination (degrees): ")), 0, np.pi),
            # Inclination, degrees
            "RAAN": convert_and_range_angle(float(input("Enter RAAN (degrees): ")), -np.pi, np.pi),  # RAAN, degrees
            "omega": convert_and_range_angle(float(input("Enter argument of perigee (degrees): ")), -np.pi, np.pi),
            # Argument of perigee, degrees
            "nu": convert_and_range_angle(float(input("Enter true anomaly (degrees): ")), -np.pi, np.pi),
            # True anomaly, degrees
            "S": float(input("Enter drag surface area (m^2): ")),  # Cross-sectional area in m^2
            "m0": float(input("Enter dry mass of the satellite (kg): ")),  # Dry mass of the satellite, kg
            "mp": float(input("Enter initial propellant mass (kg): ")),  # Initial propellant mass in kg
            "F": float(input("Enter Thrust (N): ")),  # Thrust, N
            "Isp": float(input("Enter Specific impulse (s): ")),  # Specific impulse, s
            "Cd": float(input("Enter Drag coefficient (unitless): ")),  # Drag coefficient
            "Initial_epoch": datetime.strptime(input("Enter the initial epoch (YYYY-MM-DDTHH:MM:SSZ): "),
                                               "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc),
            "burn_flag": 0,  # Assuming that no satellite is thrusting initially
            "num_sat": num_sat  # Satellite number
        }

        all_main_data.append(main_data)

    if use_parallel:
        with Pool() as pool:
            results_main = pool.map(run_simulation, all_main_data)
            all_results_main.append(results_main)
    else:
        results_main = [run_simulation(all_main_data[num_sat]) for num_sat in range(n)]
        all_results_main.append(results_main)
        # print(all_results_main)


def get_custom_threads(custom_threads=None):
    num_threads = custom_threads if custom_threads else multiprocessing.cpu_count()
    print(f"Using {num_threads} threads.")
    return num_threads


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    num_threads = get_custom_threads(4)  # Manually set to use 4 threads

    main()