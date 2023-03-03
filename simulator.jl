"""
Defining a struct for an airplane object
"""
mutable struct Airplane
    #Attitude Characteristics
    x::Float64       # distance of airplane from runway (m)
    y::Float64       # altitude of airplane from the runway (m)
    th::Float64      # pitch of the airplane (rad), horizontal is 0
    power::Float64   # throttle setting

    #Flight Dynamics
    V_air::Float64   # airspeed (m/s)
    V_vert::Float64  # vertical airspeed
    alpha::Float64   # angle of path (rad), horizontal is 0

end

"""
Defining a dynamics model via a function that updates the Airplane model
"""
function update!(model::Airplane, th', power', dt)

    #Constants
    const m = 1157          # mass of airplane (kg)
    const A = 16.17         # wing area (m^2)
    const th_max = 20*pi/360    # max pitch (rad)
    const th_min = -20*pi/360   # max pitch (rad)
    const AOI = 1.5*pi/360      # angle of incidence (wing angle of attack, offset from pitch)


    # Access existing airplane values from previous step
    x, y, th, power, V_air, alpha = model.x, model.y, model.power, model.V_air, model.alpha

    # Updating values
    power = power'
    th = th'

    # Calculate new V_air
    
    # Calculate new V_vert

    # Calculate new AOD
    alpha = asin(-V_vert/V_air)

    return model
end

"""
Defining a function to calculate the drag coefficient, given pitch.
"""
function Dcoeff(th)
    return 0.0178*exp(0.139*th)
end

"""
Defining a function to calculate the lift coefficient, given pitch.
"""
function Lcoeff(th)
    return 0.136*(th) - 0.0413*(th^2) + 0.01*(th^3) - 1.01*(10^-3)*(th^4) + 4.59*(10^-5)*(th^5) - 7.69*(10^-6)*(th^6)
end