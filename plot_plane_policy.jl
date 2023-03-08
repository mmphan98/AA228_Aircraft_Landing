"""
This function plots a potential path of the plane using the policy generated from 
the Q-Learning algorithm
"""

using CSV
using DataFrames
using Plots
include("simulator.jl")
include("state_action_space.jl")
include("data_generation.jl")


function plot_policy(inputfilename)
    inprefix = "E:\\Documents\\2023\\Winter 2023\\Decision Making Under Uncertainty\\AA228_Aircraft_Landing\\policy\\"
    inputpath = string(inprefix, inputfilename)

    policy = CSV.read(inputpath, DataFrame; header=false)

    C172 = Airplane(-4500, 300, 0.00, 150, 50, -0.0525)
    x = []
    y = []
    v = []
    th = []
    al = []


    while (sim_valid(C172))
        append!(x,C172.x)
        append!(y,C172.y)
        append!(th,C172.th)
        append!(v,C172.V_air)
        append!(al,C172.alpha)

        state_idx = find_state_idx(C172)
        action = policy[state_idx, 1]

        # Adjust pitch and power setting
        if action == 1 || action == 4 || action == 7
            th = C172.th + 0.005 #approx 0.25deg adjustments
        elseif action == 3 || action == 6 || action == 9
            th = C172.th - 0.005 #approx 0.25deg adjustments
        end

        if action == 1 || action == 2 || action == 3
            power = C172.power + 10
        elseif action == 7 || action == 8 || action == 9
            power = C172.power - 10
        end

        #Update the airplane model
        update_Airplane!(C172, th, power)

    end

    plot(x,y)

end

"""
Run plotting
"""
inputfilename = "landing4.policy";
# @time plot_policy(inputfilename)


inprefix = "E:\\Documents\\2023\\Winter 2023\\Decision Making Under Uncertainty\\AA228_Aircraft_Landing\\policy\\"
inputpath = string(inprefix, inputfilename)

policy = CSV.read(inputpath, DataFrame; header=false)

C172 = Airplane(-4500, 300, 0.00, 150, 50, -0.0525)
state_idx = find_state_idx(C172)
action = policy[state_idx, 1]

