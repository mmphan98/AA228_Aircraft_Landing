"""
This data_generation.jl file builds an a baseline dataset for feeding into the 
landing_QLearning file
"""

using Printf
using Random
using CSV
using DataFrames
include("simulator.jl")
include("state_action_space.jl")

# FOR FILE EXPORT
const savepath = "E:\\Documents\\2023\\Winter 2023\\Decision Making Under Uncertainty\\AA228_Aircraft_Landing\\data\\test_dataset.csv"

""" 
Reward Model
    -1 for every time step
    +10 for landing
    -1 for every input change
    -100 for stalling (velocity too low, <25m/s)
    -100 for crashing (velocity too high at ground level, >34m/s, )

This function calculates the reward for a give state and action, R(s,a)

"""
function calc_Reward(model::Airplane, action)
    reward = 0

    # Negative reward for each time step
    reward -= 1 

    # Negative reward for each change in pitch or power
    if (action == 1 || action == 3 || action == 7 || action == 9)
        reward -= 2 #pitch/power not same
    elseif (action == 2 || action == 4 || action == 6 || action == 8)
        reward -= 1 #pitch or power not same
    end

    # Negative reward for stalling
    if model.V_air < stall_speed + V_air_step
        reward -= 100
    end

    # Negative reward for crashing on landing
    if model.x > -x_step && model.y < y_step #at landing spot
        if model.V_air > landing_speed #overall airspeed is too fast
            reward -= 100 
        elseif abs(model.V_air*sin(model.alpha)) > landing_Vspeed_buffer #verticle component of airspeed is too large
            reward -= 100
        else
            reward += 200 #successful landing
        end
    end

    return reward

end

""" 
This function calculates the state space index for a given model state

"""
function find_state_idx(model::Airplane)
    x_idx = trunc(Int, floor((model.x - x_min)/x_step) + 1)
    y_idx = trunc(Int, floor((model.y - y_min)/y_step) + 1)
    V_air_idx = trunc(Int, floor((model.V_air -  V_air_min)/V_air_step) + 1)
    alpha_idx = trunc(Int, floor((model.alpha - alpha_min)/alpha_step) + 1)
    return S[x_idx, y_idx, V_air_idx, alpha_idx]
end

""" 
This function determines wheather the simulation is still valid based on factors listed. 
Is it is not valid, it will return false, meaning the simulation will terminate when:
    x > 0               (in the last x bucket)
    y > y_max           (above the max altitude)
    y < 0               (below the ground)
    V_air > V_air_max   (exceeding speed limit)
    V_air < V_air_min   (stalled)
    th > th_max         (can't pitch up any further)
    th < th_p_min       (can't pitch down any further)
    power > power_max   (can't add more power)
    power < power_min   (can't reduce power further)
    alpha > alpha_max   (climb is too steep)
    alpha < alpha_min   (descemt is too steep)
"""
function sim_valid(model::Airplane)
    # Position parameters
    if model.x > 0 || model.y > y_max || model.y < 0
        # @printf("Position out of bounds \n")
        return false
    # Airspeed parameters
    elseif model.V_air > V_air_max || model.V_air < V_air_min
        # @printf("Speed out of bounds \n")
        return false
    # Pitch control parameters
    elseif model.th > th_max || model.th < th_min
        # @printf("Pitch out of bounds \n")
        return false
    # Power control parameters
    elseif model.power > power_max || model.power < power_min
        # @printf("Power out of bounds \n")
        return false
    # Flight Path angle
    elseif model.alpha > alpha_max || model.alpha < alpha_min
        # @printf("Path out of bounds \n")
        return false
    else
        return true
    end
end

""" 
FOR REFERENCE: 
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

    Definition of action space
    6 possible actions
    pitch up, same, down
    throttle up, same, down

"""

# Generating random data for QLearning
dataset = Matrix{Int64}(undef, 0, 4)
const iter = 10000

for i in 1:iter
    #C172 = Airplane(-4500, 300, 0.00, 150, 50, -0.0525)
    C172 = Airplane(rand(-4500:0), rand(0:300), rand(-1745:1745)/10000, rand(20:200), rand(25:60), rand(-13:7)/100)
    while (sim_valid(C172))
        # Find the current state space index
        S_idx = find_state_idx(C172)
        # Generate random action
        rand_action = rand((1:9))
        
        # Print info
        # print(C172)
        # @printf(" Action %d \n", rand_action)

        # Initiate pitch and power variables
        th = copy(C172.th)
        power = copy(C172.power)

        # Adjust pitch and power setting
        if rand_action == 1 || rand_action == 4 || rand_action == 7
            th = C172.th + 0.005 #approx 0.25deg adjustments
        elseif rand_action == 3 || rand_action == 6 || rand_action == 9
            th = C172.th - 0.005 #approx 0.25deg adjustments
        end

        if rand_action == 1 || rand_action == 2 || rand_action == 3
            power = C172.power + 10
        elseif rand_action == 7 || rand_action == 8 || rand_action == 9
            power = C172.power - 10
        end

        #Update the airplane model
        update_Airplane!(C172, th, power)

        if (sim_valid(C172))
            R = calc_Reward(C172, rand_action)
            # @printf("Reward: %d \n", R)
            new_state = find_state_idx(C172)
            new_data = [S_idx, rand_action, R, new_state]
            global dataset = [dataset; transpose(new_data)]
        # else
        #     @printf("Simulation terminated \n")
        end
    end
end

# Write dataset to a CSV
table = Tables.table(dataset)
CSV.write(savepath, table)