using Printf

include("simulator.jl")

""" 
Airplane Model

x::Float64       # distance of airplane from runway (m)
y::Float64       # altitude of airplane from the runway (m)
th::Float64      # pitch of the airplane (rad), horizontal is 0
power::Float64   # power setting, thrust ranges from [20, 200] kg? N?
V_air::Float64   # airspeed (m/s)
V_vert::Float64  # vertical airspeed, (m/s)
alpha::Float64   # angle of path (rad), horizontal is 0

"""
#               x     y    th     power V_air V_vert alpha
C172 = Airplane(-4500, 600, 0.075, 150, 40, -2.5, -0.0525)

print(C172)
@printf("\n")

update_Airplane!(C172, 0.1, 100, 1)

print(C172)
@printf("\n")