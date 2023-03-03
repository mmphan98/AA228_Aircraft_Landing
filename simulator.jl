"""
This simulator.jl file builds a dynamics model for a Cessna 172 airplane, with a mutable struct,
and update! function, that will change the struct given new power and pitch settings.
"""

#Fundamental Constants
const roh = 1.225       # density of air, STP, kg/m^3
const g = 9.81          # gravity, m/s^2

#Plane Constants
const m = 1157          # mass of airplane (kg)
const A = 16.17         # wing area (m^2)
const th_max = 0.1745   # max pitch (rad), 20 degrees
const th_min = -0.1745  # max pitch (rad), -20 degrees
const power_max = 200   # max power, thrust ranges from [20, 200] N
const power_min = 20    # min power, thrust ranges from [20, 200] N
const AOI = 0.0131      # angle of incidence (wing angle of attack, offset from pitch), 1.5 degrees

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
    V_vert::Float64  # vertical airspeed, (m/s)
    alpha::Float64   # angle of path (rad), horizontal is 0

end

"""
Defining a dynamics model via a function that updates the Airplane model
"""
function update_Airplane!(model::Airplane, th_p, power_p, dt)

    # Access existing airplane values from previous step
    x, y, th, power, V_air, V_vert, alpha = model.x, model.y, model.th, model.power, model.V_air, model.V_vert, model.alpha

    # Updating power
    if (power_p > power_max)
        model.power = power_max
    elseif (power_p < power_min)
        model.power = power_min
    else
        model.power = power_p
    end

    # Updating angle of path
    if (th_p > th_max)
        model.th = th_max
    elseif (th_p < th_min)
        model.th = th_min
    else
        model.th = th_p
    end

    # Sum of forces
    lift = Lcoeff(th + AOI) * roh * V_air^2 * A / 2
    drag = Dcoeff(th + AOI) * roh * V_air^2 * A / 2
    Fx = (power * cos(th)) - (lift * sin(th + AOI)) - (drag * cos(th + AOI))
    Fy = (-m * g) + (power * sin(th)) + (lift * cos(th + AOI)) - (drag * sin(th + AOI))

    # Calculate new V_air
    Vx = (V_air * cos(alpha)) + (Fx / m)*dt
    Vy = (V_air * sin(alpha)) + (Fy / m)*dt
    model.V_air = sqrt(Vx^2 + Vy^2)
    
    # Calculate new V_vert
    model.V_vert = Vy

    # Calculate new alpha, angle of flight path
    model.alpha = asin(V_vert/V_air)

    # Calculate new position
    model.x += Vx * abs(cos(alpha)) * dt
    model.y += Vy * abs(sin(alpha)) * dt

    return model
end

"""
Defining a function to calculate the drag coefficient, given pitch. Exponential fit
"""
function Dcoeff(th)
    return 0.0178*exp(0.139*th)
end

"""
Defining a function to calculate the lift coefficient, given pitch. Polynomial fit of degree 6
"""
function Lcoeff(th)
    return 0.136*(th) - 0.0413*(th^2) + 0.01*(th^3) - 1.01*(10^-3)*(th^4) + 4.59*(10^-5)*(th^5) - 7.69*(10^-6)*(th^6)
end