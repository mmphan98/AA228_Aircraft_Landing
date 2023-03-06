"""
This data_generation.jl file builds an a baseline dataset for feeding into the 
landing_QLearning file
"""

using Printf
using Random
include("simulator.jl")
include("state_action_space.jl")

""" 
Reward Model
    -1 for every time step
    +10 for landing
    -1 for every input change
    -100 for stalling (velocity too low, <25m/s)
    -100 for crashing (velocity too high at ground level, >34m/s, )

"""
function calc_Reward(model::Airplane, action)
    reward = 0

    # Negative reward for each time step
    reward -= 1 

    # Negative reward for each change in pitch or power
    if (action == 1 || action == 3 || action == 7 || action == 9)
        reward -= 2 #pitch/power not same
    elseif (action == 2 || action = 4 || action == 6 || action == 8)
        reward -= 1 #pitch or power not same
    end

    # Negative reward for stalling
    if model.V_air < stall_speed
        reward -= 100
    end

    # Negative reward for crashing
    #if model.y < 

end

""" 
Airplane Model
    #Attitude Characteristics
    x::Float64       # distance of airplane from runway (m)
    y::Float64       # altitude of airplane from the runway (m)
    th::Float64      # pitch of the airplane (rad), horizontal is 0
    power::Float64   # power setting

    #Flight Dynamics
    V_air::Float64   # airspeed (m/s)
    alpha::Float64   # angle of path (rad), horizontal is 0
                          x     y    th     power V_air alpha
    Ex: C172 = Airplane(-4500, 600, 0.075, 150, 40, -0.0525)
"""

# Generating random data for QLearning
const iter = 100
rand_action = rand(1:9, iter)

for i in 1:iter
    
end
print(rand_action)