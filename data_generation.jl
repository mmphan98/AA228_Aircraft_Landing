"""
This data_generation.jl file builds an a baseline dataset for feeding into the 
landing_QLearning file
"""

using Printf
using Random
using CSV
using DataFrames
include("state_action_space.jl")

# FOR FILE EXPORT --------------------------------------------------------------------------------------------------change file here
const landing_reward = 5000

""" 
Reward Model
    -1 for every time step
    +200 for landing
    -1 for every input change
    -100 for stalling (velocity too low, <25m/s)
    -100 for crashing (velocity too high at ground level, >34m/s, )
    -100 for any parameters out of bounds

This function calculates the reward for a give state and action, R(s,a)
"""
function calc_Reward(model::Airplane, action)
    if !sim_valid(model)
        # Negative reward if out of bounds of simulation
        reward = -1000
        return reward
    end

    # Initiate reward
    reward = 0

    # Negative reward for each change in pitch or power
    if (action == 1 || action == 3 || action == 7 || action == 9)
        reward -= 20 #pitch/power not same
    elseif (action == 2 || action == 4 || action == 6 || action == 8)
        reward -= 10 #pitch or power not same
    end

    # Negative reward for stalling
    if model.V_air < stall_speed + V_air_step
        reward -= 1000
        return reward
    end

    # Positive reward for getting closer to the landing zone
    max_distant = sqrt(x_min^2 + y_max^2)
    model_distant = sqrt(model.x^2 + model.y^2)
    reward += trunc(Int, 120 * (max_distant - model_distant) / max_distant)

    # Negative reward for deviating from optimal alpha
    target_alpha = -0.0525
    reward -= trunc(Int, 60 * abs((target_alpha - model.alpha) / target_alpha))

    # Negative reward for crashing on landing
    if model.x > -2*x_step && model.y < y_step #at landing spot
        if model.V_air > landing_speed #overall airspeed is too fast
            reward -= 500 
        elseif abs(model.V_air*sin(model.alpha)) > landing_Vspeed_buffer #verticle component of airspeed is too large
            reward -= 500
        else
            reward += landing_reward #successful landing
            model.landed = true
        end
    end

    return reward

end

"""
This function calculates the change in power and pitch for a given action index
"""
function action_pitch_power_result(model::Airplane, action)
    # Initiate pitch and power variables
    th = copy(model.th)
    power = copy(model.power)

    # Adjust pitch and power setting
    if action == 1 || action == 4 || action == 7
        th = model.th + 0.01 #approx 0.25deg adjustments
    elseif action == 3 || action == 6 || action == 9
        th = model.th - 0.01 #approx 0.25deg adjustments
    end

    if action == 1 || action == 2 || action == 3
        power = model.power + 10
    elseif action == 7 || action == 8 || action == 9
        power = model.power - 10
    end

    return th, power
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
    elseif model.landed == true
        # @printf("Plane landed \n")
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

""" 
Generate random exploration data for Q-Learning
"""
function explore_dataset(dataset, iter)

    for i in 1:iter

        if i < iter/4
            C172 = Airplane(x_min, y_max, 0.00, 150, 50, -0.0525, false)
        elseif i < 2*iter/4
            C172 = Airplane(3*x_min/4, 3*y_max/4, 0.00, 150, 50, -0.0525, false)
        elseif i < 3*iter/4
            C172 = Airplane(2*x_min/4, 2*y_max/4, 0.00, 150, 50, -0.0525, false)
        elseif i < iter - 10
            C172 = Airplane(1*x_min/4, 1*y_max/4, 0.00, 150, 50, -0.0525, false)
        else
            C172 = Airplane(-x_step+1, y_step-1, 0.10, 20, 32, 0, false)
        end

        while (sim_valid(C172))
            # Find the current state space index
            S_idx = find_state_idx(C172)

            # Generate random action
            rand_action = rand((1:A_size))
            th, power = action_pitch_power_result(C172, rand_action)

            # Print info
            # print(C172)
            # @printf(" Action %d \n", rand_action)        

            #Update the airplane model
            update_Airplane!(C172, th, power)

            #Update dataset
            R = calc_Reward(C172, rand_action)
            # @printf("Reward: %d \n", R)
            if (sim_valid(C172) && C172.landed == false)
                new_state = find_state_idx(C172)
                new_data = [S_idx, rand_action, R, new_state]
            else
                new_data = [S_idx, rand_action, R, S_idx]
            #     @printf("Simulation terminated \n")
            end
            dataset = [dataset; transpose(new_data)]

            # Ending if landed succesfully
            if R > landing_reward - 20
                break
            end
        end

    end
    return dataset
end

""" 
Simulate steps ahead 
"""
function simulate(C172::Airplane, h, action)

    # for i in 1:h

        # Find the current state space index
        S_idx = find_state_idx(C172)

        # Generate action
        th, power = action_pitch_power_result(C172, action) 

        #Update the airplane model
        update_Airplane!(C172, th, power)

        #Update dataset
        R = calc_Reward(C172, action)
        # @printf("Reward: %d \n", R)
        if (sim_valid(C172) && C172.landed == false)
            new_state = find_state_idx(C172)
            new_data = [S_idx, action, R, new_state]
        else
            new_data = [S_idx, action, R, S_idx]
        #     @printf("Simulation terminated \n")
        end

    # end

    return transpose(new_data)

end

"""
UNCOMMENT TO CREATE NEW DATASET
"""

# # Compute dataset
# dataset = Matrix{Int64}(undef, 0, 4)
# dataset = @time explore_dataset(dataset)

# # Write dataset to a CSV
# table = Tables.table(dataset)
# CSV.write(savepath, table)