"""
This landing_QLearning.jl file builds an exploration and exploitation model to land a Cessna 172. 
The model uses Q-learning to find the optimal policy, and and epsilon-greed approach as a means for exploration.
"""

using Printf
include("simulator.jl")
include("state_action_space.jl")

# Defining a struct for QLearning
mutable struct QLearning
    S       # state space (assumes 1:nstates)
    A       # action space (assumes 1:nactions)
    gamma   # discount
    Q       # action value function
    alpha   # learning rate
end

# Defining a function that updates the QLearning model
function update!(model::QLearning, s, a, r, s_prime)
    gamma, Q, alpha = model.gamma, model.Q, model.alpha
    Q[s,a] += alpha*(r + gamma*maximum(Q[s_prime,:]) - Q[s,a])
    return model
end

# Defining a function that calculates the Bellman Residual given two Q matrices 
function BRes(Q, Q_prev)
    return norm(findmax(Q, dims=2)[1] - findmax(Q_prev, dims=2)[1], Inf) 
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
"""
#               x     y    th     power V_air alpha
C172 = Airplane(-4500, 600, 0.075, 150, 40, -0.0525)
update_Airplane!(C172, 0.1, 100) 
print(C172)
print(C172.V_air*cos(C172.alpha))
print(C172.V_air*sin(C172.alpha))