"""
This landing_value_iteration.jl file builds an exploration and exploitation model to land a Cessna 172. 
The model uses Q-learning to find the optimal policy, and and epsilon-greed approach as a means for exploration.
"""

using Printf
include("simulator.jl")

# Epsilon Greedy

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


"""
Definition of state space
    x --> [-4500, 0] m, mapped to bins of 100m [1 ... 46]
    y --> [0, 600] m, mapped to bins of 30m [1 ... 21] 
    x,y --> mapped to [1 ... 1911]

    V_air --> [48, 120] knots, [25, 60] m/s mapped to bins of 1 m/s [1, 36]
    alpha --> [-5, 5] deg, [-0.08, 0] rad mapped to bins of 0.01 rad [1, 9]

    Total size of state space S = 312,984
"""
# Constants
const x_max = 0
const x_min = -4500
const x_step = 100

const y_max = 600
const y_min = 0
const y_step = 30

const V_air_max = 60
const V_air_min = 25
const V_air_step = 1

const alpha_min = -0.08
const alpha_max = 0
const alpha_step = 0.01

# Building the state space
x = collect(1:((x_max - x_min)/x_step) + 1)
y = collect(1:((y_max - y_min)/y_step) + 1)
V_air = collect(1:((V_air_max - V_air_min)/V_air_step) + 1)
alpha = collect(1:((alpha_max - alpha_min)/alpha_step) + 1)

S = reshape(1:(length(x)*length(y)*length(V_air)*length(alpha)), length(x), length(y), length(V_air), length(alpha))

@printf("\n State Variable: %f", S[1,1,1,1])

"""
Definition of action space
    6 possible actions
    pitch up, same, down
    throttle up, same, down
    Total size of action space A = 9
"""