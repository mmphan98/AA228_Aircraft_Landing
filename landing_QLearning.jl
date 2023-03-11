"""
This landing_QLearning.jl file builds an exploration and exploitation model to land a Cessna 172. 
The model uses Q-learning to find the optimal policy, and and epsilon-greed approach as a means for exploration.
"""

using Printf
using CSV
using DataFrames
using LinearAlgebra
include("simulator.jl")
include("state_action_space.jl")

# Defining a function that writes the policy to the specified file output path
function writePolicy(Q, path)
    open(path, "w") do io
        for row in 1:size(Q, 1)
            if findmax(Q[row,:])[1] == 0
                @printf(io, "%d\n", 5)
            else
                @printf(io, "%d\n", findmax(Q[row,:])[2])
            end
        end
    end
end

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

 # Defining the function for Q-Learning
 function compute(infile, outfile, space, epsilon)
    
    inprefix = "E:\\Documents\\2023\\Winter 2023\\Decision Making Under Uncertainty\\AA228_Aircraft_Landing\\data\\"
    outprefix = "E:\\Documents\\2023\\Winter 2023\\Decision Making Under Uncertainty\\AA228_Aircraft_Landing\\policy\\"
    inputpath = string(inprefix, infile)
    outputpath = string(outprefix, outfile)
    
    # Read in CSV file
    df = CSV.read(inputpath, DataFrame; header = true)
    
    # Define the Q-Learning Model
    C172Model_S = collect(1:length(space))
    C172Model_A = collect(1:length(A))
    C172Model_gamma = 0.95
    C172Model_Q = zeros(length(C172Model_S),length(A))
    C172Model_alpha = 0.1
    C172Model = QLearning(C172Model_S, C172Model_A, C172Model_gamma, C172Model_Q, C172Model_alpha)

    # Iterate with Q-Learning
    Q_prev = copy(C172Model.Q)
    Q_new = []
    delta = Inf # for Bellman Residual
    iter = 0
    error = 1E-03
    while delta > error
        # Updating model based off data
        for i in 1:length(df.s)
            update!(C172Model, df.s[i], df.a[i], df.r[i], df.sp[i])
        end
        #Finding Bellman Residual
        Q_new = copy(C172Model.Q)
        delta = BRes(Q_new, Q_prev)
        Q_prev = copy(Q_new)
        iter = iter + 1
    end
    @printf("Iterations with delta = %e: %d \n", error, iter)

    # Output to .policy file
    writePolicy(Q_new, outputpath)
end





"""
UNCOMMENT TO CREATE NEW DATASET
"""

# # Compute dataset
# dataset = Matrix{Int64}(undef, 0, 4)
# dataset = explore_dataset(dataset)

# # Write dataset to a CSV
# table = Tables.table(dataset)
# CSV.write(savepath, table)

"""
Run the Q-Learning algorithm to obtain the optimal policy
"""
inputfilename = "test_dataset11.csv";
outputfilename = "landing11.policy";
space = S
epsilon = 0.8 #for epsilon greedy
@time compute(inputfilename, outputfilename, space, epsilon)