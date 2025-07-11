module SurfaceCreator

using ..ProblemSetup
using StaticArrays, DifferentialEquations

export createsurface

function createsurface(Λ, nθ, nϕ)
    θ = range(0, π, length=nθ)
    ϕ = range(0, 2π, length=nϕ)
    sphere = vec(SVector.(sin.(θ) .* cos.(ϕ)', sin.(θ) .* sin.(ϕ)', cos.(θ)))

    prob = remake(prob₀; p = Λ)
    output_func(sol, i) = (sol.t[end], sol.retcode == :Terminated)
    prob_func(p, i, repeat) = remake(p; u0 = SVector(sphere[i]..., Z₀..., Y₀...))
    ensembleprob = EnsembleProblem(prob, prob_func=prob_func, output_func=output_func, safetycopy=false)

    sols = solve(ensembleprob, DP5(), EnsembleThreads(); trajectories=nθ*nϕ, settings...)
    Tmat = reshape(first.(sols.u), nθ, nϕ)
    coords = Tmat .* reshape(sphere, nθ, nϕ)
    x, y, z = map(i -> getindex.(coords, i), 1:3)    
    return Tmat, x, y, z
end

end