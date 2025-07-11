using StaticArrays, LinearAlgebra, DifferentialEquations, BenchmarkTools, Random

println("Setting up benchmark suite...")

Λ = SVector{3,Float64}(1.2, 1, 0.8)
Z₀ = SMatrix{3,3,Float64}(I)
Y₀ = zeros(9)

nθ, nϕ = 30, 60
θ_range = range(0, π, length=nθ)
ϕ_range = range(0, 2π; length=nϕ)

@inline hat(v) = @SMatrix [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
@inline adj(X) = SMatrix{3,3}(cross(X[2,:], X[3,:])..., cross(X[3,:], X[1,:])..., cross(X[1,:], X[2,:])...)'
@inline unitvector(θ, ϕ) = @SVector [sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ)]
@inline unpackX(X) = SVector{3,Float64}(X[1], X[2], X[3]), SMatrix{3,3}(ntuple(i -> X[3+i], 9)), SMatrix{3,3}(ntuple(i -> X[12+i], 9))

function F_oop(X, Λ, t)
    u, Z, Y = unpackX(X)
    u̇ = (hat(Λ .* u) * u) ./ Λ
    Ż = (hat(u) * (Λ .* Z)) - (hat(Λ .* u) * Z)
    Ẏ = Z - (hat(u) * Y)
    return SVector{21,Float64}(u̇..., Ż..., Ẏ...)
end

function F_ip!(dX, X, Λ, t)
    u, Z, Y = unpackX(X)
    u̇ = (hat(Λ .* u) * u) ./ Λ
    Ż = (hat(u) * (Λ .* Z)) - (hat(Λ .* u) * Z)
    Ẏ = Z - (hat(u) * Y)
    dX .= SVector{21,Float64}(u̇..., Ż..., Ẏ...)
    return nothing
end

condition(out, X, t, integrator) = begin
    u, Z, Y = unpackX(X)
    Ẏ = Z - (hat(u) * Y)
    out[1] = det(Y)
    out[2] = sum(adj(Y) .* Ẏ)
end
affect_pos!(integ, idx) = terminate!(integ)
affect_neg!(integ, idx) = idx == 1 ? terminate!(integ) : nothing
fzfm = VectorContinuousCallback(condition, affect_pos!, affect_neg!, 2, interp_points=20)
settings = (callback=fzfm, reltol=1e-5, abstol=1e-5, save_everystep=false, save_start=false, save_end=true)

u₀_placeholder = SVector{3,Float64}(1.0, 0.0, 0.0)
X₀ = SVector{21,Float64}(u₀_placeholder..., Z₀..., Y₀...)
sphere = [unitvector(θ, ϕ) for θ in θ_range, ϕ in ϕ_range]
output_func(sol, i) = (sol.t[end], sol.retcode == :Terminated)

prob_oop = ODEProblem(F_oop, X₀, (0.0, 15.0), Λ)
function prob_func_oop(prob, i, repeat)
    u₀ = sphere[i]
    remake(prob; u0=SVector(u₀..., Z₀..., Y₀...))
end
eprob_oop = EnsembleProblem(prob_oop, prob_func=prob_func_oop, output_func=output_func, safetycopy=false)

prob_ip = ODEProblem(F_ip!, X₀, (0.0, 15.0), Λ)
function prob_func_ip(prob, i, repeat)
    u₀ = sphere[i]
    remake(prob; u0=MVector(u₀..., Z₀..., Y₀...))
end
eprob_ip = EnsembleProblem(prob_ip, prob_func=prob_func_ip, output_func=output_func, safetycopy=false)

suite = BenchmarkGroup()
suite["Out-of-Place (StaticArrays)"] = @benchmarkable solve($eprob_oop, DP5(), EnsembleThreads(); trajectories=$nθ*$nϕ, $settings...)
suite["In-Place (Mutable)"] = @benchmarkable solve($eprob_ip, DP5(), EnsembleThreads(); trajectories=$nθ*$nϕ, $settings...)

println("Warming up...")
warmup(suite)
println("Running benchmarks... (This may take a few minutes)")
results = run(suite, verbose = true, seconds = 60)

println("\n--- Benchmark Results ---")
display(results)

oop_time = median(results["Out-of-Place (StaticArrays)"].times)
ip_time = median(results["In-Place (Mutable)"].times)

println("\n--- Conclusion ---")
if oop_time < ip_time
    println("Out-of-Place was faster, confirming your findings.")
    ratio = ip_time / oop_time
    println("The Out-of-Place version is approximately ", round(ratio, digits=2)," times faster for this problem.\n")
else
    println("In-Place was faster.")
    ratio = oop_time / ip_time
    println("The In-Place version is approximately ", round(ratio, digits=2)," times faster for this problem.\n")
end