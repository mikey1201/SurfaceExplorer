module ProblemSetup

using StaticArrays, LinearAlgebra, DifferentialEquations

export F, prob₀, settings, X₀, hat, unpackX
#Initial conditions
const Z₀ = SMatrix{3,3,Float64}(I)
const Y₀ = @SVector zeros(9)
#adj : X -> adjugate(X)
@inline adj(X) = @SMatrix [ X[5]*X[9]-X[6]*X[8] X[3]*X[8]-X[2]*X[9] X[2]*X[6]-X[3]*X[5];
                            X[6]*X[7]-X[4]*X[9] X[1]*X[9]-X[3]*X[7] X[3]*X[4]-X[1]*X[6];
                            X[4]*X[8]-X[5]*X[7] X[2]*X[7]-X[1]*X[8] X[1]*X[5]-X[2]*X[4]]
#hat : v -> v̂
@inline hat(v) = @SMatrix [    0  -v[3]  v[2]; 
                             v[3]    0  -v[1];
                            -v[2]  v[1]    0 ]                                                                    
@inline unpackX(X) = SVector{3,Float64}(X[1], X[2], X[3]),  #u 
                     SMatrix{3,3}(ntuple(i -> X[3+i], 9)),  #Z
                     SMatrix{3,3}(ntuple(i -> X[12+i], 9))  #Y
#Differential equation function
function F(X, Λ, t) #Benchmark Results: Out-of-Place (SVector) => Trial(6.039 ms), In-Place (MVector) => Trial(7.095 ms)
    u, Z, Y = unpackX(X)
    u̇ = (hat(Λ .* u) * u) ./ Λ                  #u̇ = ad_u⃰(u)
    Ż = (hat(u) * (Λ .* Z)) - (hat(Λ .* u) * Z) #Ż = ad_u⃰(Z) + ad_Z⃰(u)
    Ẏ = Z - (hat(u) * Y)                        #Ẏ = Z - ad_u(Y)
    return SVector{21,Float64}(u̇..., Ż..., Ẏ...)
end 
#The callback will terminate the solver when either of the out variables are equal to 0
condition(out, X, t, integrator) = begin
    u, Z, Y = unpackX(X)
    Ẏ = Z - (hat(u) * Y)
    out[1] = det(Y)
    out[2] = sum(adj(Y) .* Ẏ)   #d/dt(det(Y)) == 0  here we solve for it by taking the Frobenius inner product of adj(Y) and Ẏ
end                             #this computation involves near singular matrices and forces us to use tighter tolerances / more interpolants to obtain accurate results
affect_pos!(integ, idx) = terminate!(integ)                                                                                     #High interp points allows us to lower the tolerances of the solver
affect_neg!(integ, idx) = idx == 1 ? terminate!(integ) : nothing                                                                #this is desirable because interp points are cheaper than precision
const fzfm = VectorContinuousCallback(condition, affect_pos!, affect_neg!, 2, interp_points=20, save_positions=(false,false))   #The combo of the DP5() solver, interp_points=20, and tol=1e-5
const settings = (callback=fzfm, reltol=1e-5, abstol=1e-5, dense=false, save_everystep=false, save_start=false, save_end=true)  #is a sweet spot for good accuracy and acceptable speed

const u₀ = SVector{3,Float64}(1.0, 0.0, 0.0)
const X₀ = SVector{21,Float64}(u₀..., Z₀..., Y₀...)
const prob₀ = ODEProblem(F, X₀, (0.0, 15.0), [1.2, 1.0, 0.8])

end