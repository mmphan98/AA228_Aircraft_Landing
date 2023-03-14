"""
This function plots a potential path of the plane using the policy generated from 
the Q-Learning algorithm
"""

using CSV
using DataFrames
using Plots
include("data_generation.jl")


function plot_policy(inputfilename, outputfilename)
    inprefix = "E:\\Documents\\2023\\Winter 2023\\Decision Making Under Uncertainty\\AA228_Aircraft_Landing\\policy\\"
    outprefix = "E:\\Documents\\2023\\Winter 2023\\Decision Making Under Uncertainty\\AA228_Aircraft_Landing\\plots\\"
    
    # inprefix = "policy/"
    # outprefix = "plots/"

    inputpath = string(inprefix, inputfilename)
    outputpath = string(outprefix, outputfilename)

    policy = CSV.read(inputpath, DataFrame; header = false)

    C172 = Airplane(x_min, y_max, 0.00, 150, 50, -0.0525, false)
    plot_x = Vector{Float64}()
    plot_y = Vector{Float64}()
    plot_th = Vector{Float64}()
    plot_p = Vector{Float64}()
    plot_v = Vector{Float64}()
    plot_al = Vector{Float64}()

    while (sim_valid(C172))
        push!(plot_x, C172.x)
        push!(plot_y, C172.y)
        push!(plot_th, C172.th)
        push!(plot_p, C172.power)
        push!(plot_v, C172.V_air)
        push!(plot_al, C172.alpha)

        state_idx = find_state_idx(C172)
        action = policy[state_idx, 1]
        th, power = action_pitch_power_result(C172, action)

        #Update the airplane model
        update_Airplane!(C172, th, power)
        if C172.landed == true
            @printf("PLANE LANDED!!!\n")
        end
    end

    p1 = plot(plot_x, plot_y, title="x v. y", seriestype=:scatter)
    p2 = plot(plot_x, plot_th, title="x v. th", seriestype=:scatter)
    p3 = plot(plot_x, plot_p, title="x v. power", seriestype=:scatter)
    p4 = plot(plot_x, plot_v, title="x v. velocity", seriestype=:scatter)
    p5 = plot(plot_x, plot_al, title="x v. alpha", seriestype=:scatter)
    p6 = plot(plot_th, plot_v, title="th v. velocity", seriestype=:scatter)
    plot(p1, p2, p3, p4, p5, p6, layout=(3,3), legend=false)
    savefig(outputpath) 

end

"""
UNCOMMENT TO CREATE NEW PLOTS

Run plotting
"""
inputfilename = "landing13.policy";
outputfilename = "testplot13.png";
@time plot_policy(inputfilename, outputfilename)