include("simulator.jl")


"""
Definition of state space
    x --> [-4500, 0] m, mapped to bins of 100m [1 ... 46]
    y --> [0, 600] m, mapped to bins of 30m [1 ... 21] 
    x,y --> mapped to [1 ... 1911]

    V_air --> [48, 120] knots, [25, 60] m/s mapped to bins of 1 m/s [1, 36]
    alpha --> [-5, 5] deg, [-0.08, 0.02] rad mapped to bins of 0.01 rad [1, 11]

    Total size of state space S = 382,536
"""
# Constants
const x_max = 0
const x_min = -4500
const x_step = 100

const y_max = 600
const y_min = 0
const y_step = 30

const V_air_max = 60
const V_air_min = stall_speed
const V_air_step = 1

const alpha_min = -0.08
const alpha_max = 0.02
const alpha_step = 0.01

# Building the state space
x = collect(1:((x_max - x_min)/x_step) + 1)
y = collect(1:((y_max - y_min)/y_step) + 1)
V_air = collect(1:((V_air_max - V_air_min)/V_air_step) + 1)
alpha = collect(1:((alpha_max - alpha_min)/alpha_step) + 1)
S_size = length(x)*length(y)*length(V_air)*length(alpha)

S = reshape(1:S_size, length(x), length(y), length(V_air), length(alpha))

"""
Definition of action space
    6 possible actions
    pitch up, same, down
    throttle up, same, down
    Total size of action space A = 9
"""
# Constants
const th_p_max = 2
const th_p_min = 0
const th_p_step = 1

const power_p_max = 2
const power_p_min = 0
const power_p_step = 1

# Building the action space
th_p = collect(1:((th_p_max - th_p_min)/th_p_step) + 1)
power_p = collect(1:((power_p_max - power_p_min)/power_p_step) + 1)
A_size = length(th_p)*length(power_p)

A = reshape(1:A_size, length(th_p), length(power_p))