# LEO Constellation End-Of-Life Mission Optimization Tool

## Overview
This Python script provides a reusable tool for end-of-life mission planning and enabling controlled deorbit of
LEO satellites while maximizing fuel consumption to ensure safe and secured end-of-life disposal.
This tool uses simplified dynamics with advanced orbital mechanics and numerical integration techniques.
The script calculates optimal burn sequences to lower satellite orbits and manage propellant
consumption during atmospheric reentry.

## Features
- Simulate deorbiting for multiple satellites simultaneously
- Compute optimal burn sequences
- Model atmospheric drag and thrust effects
- Visualize orbital dynamics and propellant consumption

## Prerequisites

### Dependencies
- Python 3.8+
- NumPy
- SciPy
- Poliastro
- Astropy
- Matplotlib
- Multiprocessing

### Installation
```bash
pip install numpy scipy poliastro astropy matplotlib
```

## Usage

### Running the Simulation
```bash
python constellation_deorbit_optimization.py
```

### Running the simulation with higher precision and improved dynamics
```bash
python constellation_deorbit_optimization_precision.py
```

### Input Parameters
When prompted, provide the following for each satellite:
1. Semi-major axis (meters)
2. Eccentricity
3. Inclination (degrees)
4. Right Ascension of Ascending Node (RAAN, degrees)
5. Argument of Perigee (degrees)
6. True Anomaly (degrees)
7. Drag surface area (m²)
8. Dry mass (kg)
9. Initial propellant mass (kg)
10. Thrust (N)
11. Specific impulse (seconds)
12. Drag coefficient
13. Initial mission epoch (YYYY-MM-DDTHH:MM:SSZ format)

### Parallel Processing
- Option to enable/disable parallel processing
- Configurable number of threads

## Methodology
The simulation uses:
- Gauss planetary equations
- Runge-Kutta 45 numerical integration
- Atmospheric density model
- Drag and thrust acceleration calculations

## Outputs
- Optimal burn start times
- Remaining propellant
- Visualizations:
  * Satellite position vs time
  * Altitude vs time
  * Propellant consumption vs time

## Prominent Functions
- atmospheric_density() : Calculate atmospheric density at a given altitude using an exponential model.
- drag_acceleration() : Compute drag acceleration based on satellite parameters and atmospheric conditions.
- compute_acceleration() : Compute acceleration vector in Radial-Tangential-Normal (RTN) frame.
- gauss_planetary_equations() : Compute the derivatives of orbital elements using Gauss planetary equations.
- solve_ode() : Propagate satellite orbit using Gauss planetary equations and Runge-Kutta 45 numerical integrator.
- optimize_burn_sequence() : Optimize the burn sequence to achieve deorbiting with maximum propellant usage.
- run_simulation() : Run a complete deorbiting simulation for a single satellite. Processes satellite parameters, performs orbital propagation,
        optimizes burn sequence, and generates results.
- plot_results() : Plot position vs time, altitude vs time, and propellant consumption vs time
- main() : Main program to simulate controlled deorbiting for multiple satellites.

## Limitations
- Simplified dynamics
- Simplified atmospheric and gravitational models
- Assumes constant specific impulse and perfect thrust
- Limited to 30-day simulation window

## Contact
Dr. Rachit Bhatia
rachitbhatia31@gmail.com
