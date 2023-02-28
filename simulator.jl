"""
Defining a struct for an airplane object
"""
mutable struct Airplane
    x       # distance of airplane from runway (m)
    y       # altitude of airplane from the runway (m)
    th      # pitch of the airplane (rad), horizontal is 0
    power   # throttle setting [0, 10]

    V_air   # airspeed (m/s)
    AOD     # angle of descent from runway (rad), horizontal is 0
end

"""
Defining a dynamics model via a function that updates the Airplane model
"""
function update!(model::Airplane, th', power', dt)
    #Constants
    Cd = 0.001      # drag coefficient
    Cl = 0.001      # lift coefficient
    weight = 1      # weight of airplane

    # Access existing airplane values from previous step
    x, y, th, power, V_air, AOD = model.x, model.y, model.power, model.V_air, model.AOD

    # Updating values
    power = power'
    th = th'

    # Calculate new V_air
    V_air += power / (Cd * th)

    # Calculate new AOD
    AOD = (weight - (Cl * V_air)) + th

    return model
end