"""
This landing_QLearning.jl file builds an exploration and exploitation model to land a Cessna 172. 
The model uses Q-learning to find the optimal policy, and and epsilon-greed approach as a means for exploration.
"""

using Printf
using CSV
using DataFrames
using LinearAlgebra
# include("data_generation.jl")
include("plot_plane_policy.jl")

# include("data_generation_sequential.jl")

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
 function compute(infile, outfile, space)
    
    inprefix = "E:\\Documents\\2023\\Winter 2023\\Decision Making Under Uncertainty\\AA228_Aircraft_Landing\\data\\"
    outprefix = "E:\\Documents\\2023\\Winter 2023\\Decision Making Under Uncertainty\\AA228_Aircraft_Landing\\policy\\"
    inputpath = string(inprefix, infile)
    outputpath = string(outprefix, outfile)
    
    # Read in CSV file
    df = CSV.read(inputpath, DataFrame; header = true)
    rename!(df, [:s, :a, :r, :sp])
    
    # Define the Q-Learning Model
    C172Model_S = collect(1:length(space))
    C172Model_A = collect(1:length(A))
    C172Model_gamma = 1
    C172Model_Q = zeros(length(C172Model_S),length(A))
    C172Model_alpha = 0.3
    C172Model = QLearning(C172Model_S, C172Model_A, C172Model_gamma, C172Model_Q, C172Model_alpha)

    # Iterate with Q-Learning
    Q_prev = copy(C172Model.Q)
    Q_new = []
    delta = Inf # for Bellman Residual
    iter = 0
    error = 1
    # while delta > error
    while iter < 5
        @printf("Iteration: %d, Delta: %f\n", iter, delta)
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

# Defining the function for Q-Learning with epsilon greedy
function compute_epsilon(C172Model::QLearning, dataset, h, maxIter)

    C172 = Airplane(x_min, y_max, 0.00, 150, 50, -0.0525, false)

    for i in 1:10
        epsilon = (11-i)*0.10
        iter = 0
        while !C172.landed && (iter < maxIter)

            # Get state index
            S_idx = find_state_idx(C172)

            # Determine action to take
            if (rand() < epsilon)
                action = rand((1:A_size))
            else
                action = findmax(C172Model.Q[S_idx,:])[2]
            end
        
            # Append new data (s,a,r,sp)
            dataset = [dataset; simulate(C172, h, action)]
        
            # Update Q-function with new row
            update!(C172Model, dataset[end,1], dataset[end,2], dataset[end,3], dataset[end,4])
        
            # Restart simulation if plane simulation faults
            if (!sim_valid(C172))
                C172 = Airplane(x_min, y_max, 0.00, 150, 50, -0.0525, false)
            end
        
            iter += 1
        
        end

    end

    table = Tables.table(dataset)
    CSV.write(savepath, table)

end




const savepath = "E:\\Documents\\2023\\Winter 2023\\Decision Making Under Uncertainty\\AA228_Aircraft_Landing\\data\\test_dataset15.csv"

"""
Runs an epislon greedy exploration stragety
"""
# Compute dataset with epsilon greedy
C172Model_S = collect(1:length(S))
C172Model_A = collect(1:length(A))
C172Model_gamma = 1
C172Model_Q = zeros(length(C172Model_S),length(A))
C172Model_alpha = 0.3
C172Model = QLearning(C172Model_S, C172Model_A, C172Model_gamma, C172Model_Q, C172Model_alpha)

dataset = Matrix{Int64}(undef, 0, 4)
h = 1
maxIter = 10000
@time compute_epsilon(C172Model, dataset, h, maxIter)


"""
Runs an randomized exploration strategy and generates a dataset
"""
# dataset = Matrix{Int64}(undef, 0, 4)
# iter = 50
# dataset = explore_dataset(dataset, iter)

# # Write dataset to a CSV
# table = Tables.table(dataset)
# const savepath = "E:\\Documents\\2023\\Winter 2023\\Decision Making Under Uncertainty\\AA228_Aircraft_Landing\\data\\test_dataset13.csv"
# CSV.write(savepath, table)


"""
Run the Q-Learning algorithm to obtain the optimal policy
"""
inputfilename = "test_dataset15.csv";
outputfilename = "landing15.policy";
space = S
@time compute(inputfilename, outputfilename, space)


"""
UNCOMMENT TO CREATE NEW PLOTS

Run plotting
"""
inputfilename = "landing15.policy";
outputfilename = "testplot15.png";
@time plot_policy(inputfilename, outputfilename)