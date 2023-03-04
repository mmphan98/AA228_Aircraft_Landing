"""
This simulator.jl file builds a dynamics model for a Cessna 172 airplane, with a mutable struct,
and update! function, that will change the struct given new power and pitch settings.
"""

using Random
using Distributions

#Fundamental Constants
const roh = 1.225       # density of air, STP, kg/m^3
const g = 9.81          # gravity, m/s^2

#Plane Constants
const m = 1000          # mass of airplane (kg)
const A = 16.17         # wing area (m^2)
const th_max = 0.1745   # max pitch (rad), 20 degrees
const th_min = -0.1745  # max pitch (rad), -20 degrees
const power_max = 200   # max power, thrust ranges from [20, 200] N
const power_min = 20    # min power, thrust ranges from [20, 200] N
const AOI = 0.0131      # angle of incidence (wing angle of attack, offset from pitch), 1.5 degrees

#Noise Parameters
const wind_speed = Normal(0,2)

#Model Constants
const dt = 3            # time step for the dynamics model [s]

"""
Defining a struct for an airplane object
"""
mutable struct Airplane

    #Attitude Characteristics
    x::Float64       # distance of airplane from runway (m)
    y::Float64       # altitude of airplane from the runway (m)
    th::Float64      # pitch of the airplane (rad), horizontal is 0
    power::Float64   # power setting

    #Flight Dynamics
    V_air::Float64   # airspeed (m/s)
    alpha::Float64   # angle of path (rad), horizontal is 0

end

"""
Defining a dynamics model via a function that updates the Airplane model
"""
function update_Airplane!(model::Airplane, th_p, power_p)

    # Access existing airplane values from previous step
    th, power, V_air, alpha = model.th, model.power, model.V_air, model.alpha

    # Updating power
    model.power = power_p

    # Updating angle of path
    model.th = th_p

    # Sum of forces
    lift = Lcoeff(th + AOI) * roh * V_air^2 * A / 2
    drag = Dcoeff(th + AOI) * roh * V_air^2 * A / 2
    Fx = (power * g * cos(th)) - (lift * sin(th + AOI)) - (drag * cos(th + AOI))
    Fy = (-m * g) + (power * g * sin(th)) + (lift * cos(th + AOI)) - (drag * sin(th + AOI))

    # Calculate new V_air
    Vx = (V_air * cos(alpha)) + (Fx / m)*dt
    Vy = (V_air * sin(alpha)) + (Fy / m)*dt
    # Adding Gaussian noise for wind
    Vx += rand(wind_speed,1)[1]
    Vy += rand(wind_speed,1)[1]
    model.V_air = sqrt(Vx^2 + Vy^2)

    # Calculate new alpha, angle of flight path
    model.alpha = asin(Vy/V_air)

    # Calculate new position
    model.x += Vx * abs(cos(alpha)) * dt
    model.y += Vy * abs(sin(alpha)) * dt

    return model
end

"""
Defining a function to calculate the drag coefficient, given pitch. Exponential fit vs angle of attack in deg
"""
function Dcoeff(th)
    th*=180/pi
    return 0.0178*exp(0.139*th)
end

"""
Defining a function to calculate the lift coefficient, given pitch. Polynomial fit of degree 6 vs angle of attack in deg
"""
function Lcoeff(th)
    th*=180/pi
    return 0.136*(th) - 0.0413*(th^2) + 0.01*(th^3) - 1.01*(10^-3)*(th^4) + 4.59*(10^-5)*(th^5) - 7.69*(10^-6)*(th^6)
end