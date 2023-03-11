using Printf
include("data_generation.jl")

for i in 1:100
    C172 = Airplane(-x_step+1, y_step-1, 0.10, 20, 32, 0)
    print(C172)
    @printf("\n")

    action = rand((1:9))

    th, power = action_pitch_power_result(C172, action)
    update_Airplane!(C172, th, power)

    print(C172)
    @printf("\n")

    R = calc_Reward(C172, action)
    @printf("Action: %d, Reward: %d \n", action, R)
end
