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


function plot_policy(inputfilename, outputfilename)
    inprefix = "E:\\Documents\\2023\\Winter 2023\\Decision Making Under Uncertainty\\AA228_Aircraft_Landing\\policy\\"
    outprefix = "E:\\Documents\\2023\\Winter 2023\\Decision Making Under Uncertainty\\AA228_Aircraft_Landing\\plots\\"
    inputpath = string(inprefix, inputfilename)
    outputpath = string(outprefix, outputfilename)

    policy = CSV.read(inputpath, DataFrame; header=false)

    C172 = Airplane(-4500, 300, 0.00, 150, 50, -0.0525)
    x = Vector{Float64}()
    y = Vector{Float64}()
    th = Vector{Float64}()
    p = Vector{Float64}()
    v = Vector{Float64}()
    al = Vector{Float64}()


    while (sim_valid(C172))
        push!(x,C172.x)
        push!(y,C172.y)
        push!(th,C172.th)
        push!(p,C172.power)
        push!(v,C172.V_air)
        push!(al,C172.alpha)

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
    savefig(outputpath) 

end

"""
Run plotting
"""
inputfilename = "landing4.policy";
outputfilename = "testplot4.png";
@time plot_policy(inputfilename, outputfilename)

