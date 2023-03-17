using Distributions
using Printf
# include("simulator.jl")
# include("state_action_space.jl")
include("data_generation.jl")

landed_actions = [6,5,1,6,4,9,2,4,5,5,5,1,3,6,8,8,4,1,6,8,8,1,8,3,1,1,1]
C172 = Airplane(x_min, y_max, 0.11, 150, 33, -0.0525, false)

for i in landed_actions
    # print(C172)
    # @printf("\n")

    action = i

    th, power = action_pitch_power_result(C172, action)
    update_Airplane!(C172, th, power)

    print(C172)
    @printf("\n")

    R = calc_Reward(C172, action)
    @printf("Action: %d, Reward: %d \n", action, R)
end

print(C172.landed)


#                 x     y                  V_air   alpha
#                 1     13                  26    8         
# model = Airplane(x_min, y_max, 0.00, 150, 50, -0.0525)

# x_idx = trunc(Int, floor((model.x - x_min)/x_step) + 1)
# y_idx = trunc(Int, floor((model.y - y_min)/y_step) + 1)
# V_air_idx = trunc(Int, floor((model.V_air -  V_air_min)/V_air_step) + 1)
# alpha_idx = trunc(Int, floor((model.alpha - alpha_min)/alpha_step) + 1)

# @printf("x: %d, y: %d, V_air: %d, alpha: %d\n", x_idx, y_idx, V_air_idx, alpha_idx)

# print(find_state_idx(model))
# @printf("\n")
# print(S[1, 13, 26, 8])

# print(Normal(0,0)[1])