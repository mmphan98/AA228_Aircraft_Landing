"""
This function plots a potential path of the plane using the policy generated from 
the Q-Learning algorithm
"""

using CSV
using DataFrames
using Plots


function plot_policy(inputfilename, outputfilename)
    inprefix = "policy/"
    outprefix = "plots/"

    inputpath = string(inprefix, inputfilename)
    outputpath = string(outprefix, outputfilename)

    policy = CSV.read(inputpath, DataFrame; header = false)

    # C172 = Airplane(x_min, y_max, 0.00, 150, 50, -0.0525, false)
    C172 = Airplane(x_min, y_max, 0.11, 150, 33, -0.0525, false)

    plot_x = Vector{Float64}()
    plot_y = Vector{Float64}()
    plot_th = Vector{Float64}()
    plot_p = Vector{Float64}()
    plot_v = Vector{Float64}()
    plot_al = Vector{Float64}()
    plot_r = Vector{Float64}()
    total_reward = 0

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

        # println(calc_Reward(C172, action))
        total_reward += calc_Reward(C172, action)
        push!(plot_r, total_reward)

        if C172.landed == true
            @printf("PLANE LANDED!!!\n")
        end
    end

    # Plotting in one figure
    p1 = plot(plot_x, plot_y, title="y v. x", seriestype=:scatter)
    p2 = plot(plot_x, plot_th, title="th v. x", seriestype=:scatter)
    p3 = plot(plot_x, plot_p, title="power v. x", seriestype=:scatter)
    p4 = plot(plot_x, plot_v, title="velocity v. x", seriestype=:scatter)
    p5 = plot(plot_x, plot_al, title="alpha v. x", seriestype=:scatter)
    p6 = plot(plot_x, plot_r, title="tot. reward v. x", seriestype=:scatter)
    plot(p1, p2, p3, p4, p5, p6, layout=(3,3), legend=false)
    savefig(outputpath) 

end

"""
UNCOMMENT TO CREATE NEW PLOTS

Run plotting
"""
# inputfilename = "landing14.policy";
# outputfilename = "testplot14.png";
# @time plot_policy(inputfilename, outputfilename)