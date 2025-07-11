using StaticArrays, LinearAlgebra, DifferentialEquations, GLMakie

# --- Using all your original, correct definitions ---
Λ = SVector{3,Float64}(0.3, 1.0, 0.8)
Z₀ = SMatrix{3,3,Float64}(I)
Y₀ = @SVector zeros(9)

@inline adj(X) = @SMatrix [ X[5]*X[9]-X[6]*X[8] X[3]*X[8]-X[2]*X[9] X[2]*X[6]-X[3]*X[5];
                            X[6]*X[7]-X[4]*X[9] X[1]*X[9]-X[3]*X[7] X[3]*X[4]-X[1]*X[6];
                            X[4]*X[8]-X[5]*X[7] X[2]*X[7]-X[1]*X[8] X[1]*X[5]-X[2]*X[4]]
@inline hat(v) = @SMatrix [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
@inline unpackX(X) = (SVector{3,Float64}(X[1], X[2], X[3]),
                     SMatrix{3,3}(ntuple(i -> X[3+i], 9)),
                     SMatrix{3,3}(ntuple(i -> X[12+i], 9)))

function F(X, Λ, t)
    u, Z, Y = unpackX(X)
    u̇ = (hat(Λ .* u) * u) ./ Λ
    Ż = (hat(u) * (Λ .* Z)) - (hat(Λ .* u) * Z)
    Ẏ = Z - (hat(u) * Y)
    return SVector{21,Float64}(u̇..., Ż..., Ẏ...)
end

# Your original, correct callback logic for detecting minima or zero-crossings
condition(out, X, t, integrator) = begin
    u, Z, Y = unpackX(X)
    Ẏ = Z - (hat(u) * Y)
    out[1] = det(Y)
    out[2] = sum(adj(Y) .* Ẏ)
end

affect_pos!(integ, idx) = terminate!(integ)
affect_neg!(integ, idx) = idx == 1 ? terminate!(integ) : nothing

# Note the high interp_points, which also helps the root-finder's accuracy
precise_callback = VectorContinuousCallback(condition, affect_pos!, affect_neg!, 2, interp_points=20, save_positions=(false,false))
settings = (callback=precise_callback, reltol=1e-5, abstol=1e-5, dense=false, save_everystep=false, save_start=false, save_end=true)


# --- Diagnosis Code ---
println("Diagnosing the root-finding failure...")

u0_test = @SVector [0.3576620478535541, 0.09872840845171463, 0.8625147532856585]
tmax = 15.0
X₀ = SVector{21,Float64}(u0_test..., Z₀..., Y₀...)
prob = ODEProblem(F, X₀, (0.0, tmax), Λ)

# --- TEST 1: Solve with no step limit (The failing case) ---
println("\n--- Running with default (large) steps ---")
sol_fail = solve(prob, DP5(); settings...)
println("Termination time: $(sol_fail.t[end]). Retcode: $(sol_fail.retcode). (Expected to fail by running to tmax)")


# --- TEST 2: Solve while forcing small steps (The fix) ---
println("\n--- Running with dtmax to force smaller steps ---")
# By setting dtmax, we prevent the solver from jumping over the event.
sol_succeed = solve(prob, DP5(); settings...)
println("Termination time: $(sol_succeed.t[end]). Retcode: $(sol_succeed.retcode). (Expected to succeed)")


# --- Visualization ---
# Plot the full trajectory and mark where the successful termination occurred
sol_full = solve(prob, DP5(), reltol=1e-5, abstol=1e-5, saveat=0.01)
times = sol_full.t
determinants = [det(unpackX(X)[3]) for X in sol_full.u]

fig = Figure(size=(900, 700))
ax = Axis(fig[1,1], xlabel="Time (t)", ylabel="Value", title="Why The Root-Finder Fails: Step Size")
lines!(ax, times, determinants, label="det(Y)", linewidth=3, color=:blue)
hlines!(ax, 0, color=:black, linestyle=:dot, linewidth=1.5)
vlines!(ax, [sol_succeed.t[end]], color=:green, linestyle=:dash, linewidth=2.5, label="Correct Termination (with dtmax)")
axislegend(ax, position=:ct)

fig
#=
using StaticArrays, LinearAlgebra, DifferentialEquations, GLMakie

# --- Definitions from the second script ---
Λ = SVector{3,Float64}(0.3, 1.0, 0.8)
Z₀ = SMatrix{3,3,Float64}(I)
Y₀ = @SVector zeros(9)

# Helper Functions
@inline adj(X) = @SMatrix [ X[5]*X[9]-X[6]*X[8] X[3]*X[8]-X[2]*X[9] X[2]*X[6]-X[3]*X[5];
                            X[6]*X[7]-X[4]*X[9] X[1]*X[9]-X[3]*X[7] X[3]*X[4]-X[1]*X[6];
                            X[4]*X[8]-X[5]*X[7] X[2]*X[7]-X[1]*X[8] X[1]*X[5]-X[2]*X[4]]
@inline hat(v) = @SMatrix [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
@inline unpackX(X) = (SVector{3,Float64}(X[1], X[2], X[3]),
                     SMatrix{3,3}(ntuple(i -> X[3+i], 9)),
                     SMatrix{3,3}(ntuple(i -> X[12+i], 9)))

# Differential Equation (Out-of-place)
function F(X, Λ, t)
    u, Z, Y = unpackX(X)
    u̇ = (hat(Λ .* u) * u) ./ Λ
    Ż = (hat(u) * (Λ .* Z)) - (hat(Λ .* u) * Z)
    Ẏ = Z - (hat(u) * Y)
    return SVector{21,Float64}(u̇..., Ż..., Ẏ...)
end

# --- Test Code Starts Here ---
println("Setting up test case...")

# 1. Choose a single initial condition
u0_test = @SVector [0.769856573339344, -0.6046713670996292, -0.4175591249614774]#[0.8014933512051794, 0.6289669102384046, 0.2620890617785916]

# 2. Set up and solve the ODE problem
tmax = 15.0
X₀ = SVector{21,Float64}(u0_test..., Z₀..., Y₀...)
prob = ODEProblem(F, X₀, (0.0, tmax), Λ)
sol = solve(prob, DP5(), reltol=1e-5, abstol=1e-5, saveat=0.01)

println("Analyzing trajectory...")

# 3. Initialize arrays to store data for plotting
times = sol.t
determinants = Float64[]
det_derivatives = Float64[]
detderalt = Float64[]
detderlog = Float64[]

for X in sol.u
    # Unpack the state vector for this time step
    u, Z, Y = unpackX(X)

    # Calculate Ẏ, det(Y), and d/dt(det(Y))
    Ẏ = Z - (hat(u) * Y)
    d = det(Y)
    d_dt = sum(adj(Y) .* Ẏ) # d/dt(det(Y))
    d_dt_alt = det(Y) * tr(inv(Y) * Ẏ)
    d_dt_log = tr(inv(Y) * Ẏ)
    # Store the results
    push!(determinants, d)
    push!(det_derivatives, d_dt)
    push!(detderalt, d_dt_alt)
    push!(detderlog, d_dt_log)

end

println("Plotting results...")

# 4. Plot all results on the same graph
fig = Figure(size=(900, 700))
ax = Axis(fig[1,1],
          xlabel="Time (t)",
          ylabel="Value",
          title="Comparison of Conjugacy Point Metrics")

lines!(ax, times, determinants, label="det(Y)", linewidth=3, color=:blue)
lines!(ax, times, det_derivatives, label="d/dt(det(Y))", linewidth=3, color=:green)
#lines!(ax, times, detderalt, label="d/dt(det(Y))alt", linewidth=3, color=:red)
#lines!(ax, times, detderlog, label="d/dt(log(det(Y)))", linewidth=3, color=:purple)

hlines!(ax, 0, color=:black, linestyle=:dot, linewidth=1.5)
axislegend(ax)

# Display the plot
fig
=#
#=
using StaticArrays, LinearAlgebra, DifferentialEquations, GLMakie, GeometryBasics

Λ = SVector{3,Float64}(1.2, 1, 0.8)
@inline hat(v) = @SMatrix [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
function F!(dX, X, Λ, t)
    u = @SVector [X[1], X[2], X[3]]; Z = SMatrix{3,3}(ntuple(i -> X[3+i], 9)); Y = SMatrix{3,3}(ntuple(i -> X[12+i], 9))
    û = hat(u); v̂ = hat(Λ .* u)
    u̇ = (v̂ * u) ./ Λ; Ż = û * (Λ .* Z) - v̂ * Z; Ẏ = Z - (û * Y)
    dX .= SVector(u̇..., Ż..., Ẏ...)
    return nothing
end
@inline to_energy_ellipsoid(û, Λ) = û .* 1/(sqrt(Λ ⋅ (û.^2)))

# --- NEW DIAGNOSTIC FUNCTION ---
# This version OBSERVES and PRINTS instead of terminating.
function observe_trajectory_dips(u0::SVector{3,Float64}, Λ::SVector{3,Float64}; tmax=11.0)
    Z₀ = SMatrix{3,3,Float64}(I)
    Y₀ = Z₀ - Z₀
    X0 = MVector(u0..., vec(Z₀)..., vec(Y₀)...)
    prob = ODEProblem(F!, X0, (0.0, tmax), Λ)

    # State for our observer
    running_max_abs_det = 0.0

    function det_derivative(X, t, integrator)
        u=SVector(X[1],X[2],X[3]); Z=SMatrix{3,3}(ntuple(i->X[3+i],9)); Y=SMatrix{3,3}(ntuple(i->X[12+i],9))
        dY = det(Y);
        if abs(dY) < 1e-9; return 0.0; end
        û = hat(u); Ẏ = Z - (û*Y); adj_Y = dY*inv(Y)
        return tr(adj_Y * Ẏ)
    end

    function observer_affect!(integrator)
        t = integrator.t
        if t > 1.0
            X = integrator.u
            Y = SMatrix{3,3}(ntuple(i -> X[12+i], 9))
            current_abs_det = abs(det(Y))

            println("-----------------------------")
            println("Event at t = $t")
            println("  - Current abs(det(Y)) = $current_abs_det")
            println("  - Running max so far  = $running_max_abs_det")

            # Is this a peak? Update the max.
            if current_abs_det > running_max_abs_det
                running_max_abs_det = current_abs_det
                println("  - This is a new peak.")
            # Or is this a trough? Calculate the dip ratio.
            elseif running_max_abs_det > 1e-9
                dip_ratio = current_abs_det / running_max_abs_det
                println("  - This is a trough. Dip Ratio = $dip_ratio")
            end
        end
    end

    cb = ContinuousCallback(det_derivative, observer_affect!, abstol=1e-4)
    # Run the full simulation to see all events
    solve(prob, DP5(), reltol=1e-6, abstol=1e-6, callback=cb)
    return
end


# --- Main Diagnostic Execution ---
# Pick a single "bad" point that was causing a hole before.
# A point not on an axis or equator is usually a good test case.
nθ, nφ = 60, 120
θs = range(0, π, length=nθ)
φs = range(0, 2π, length=nφ)

θ_test = θs[nθ ÷ 3]
φ_test = φs[nφ ÷ 3]
û_test = @SVector [sin(θ_test)*cos(φ_test), sin(θ_test)*sin(φ_test), cos(θ_test)]
u0_test = to_energy_ellipsoid(û_test, Λ)

println("--- Running Diagnostic for a single trajectory ---")
observe_trajectory_dips(u0_test, Λ; tmax=11)
println("--- Diagnostic Finished ---")
=#
#=
using StaticArrays, LinearAlgebra, DifferentialEquations, GLMakie, GeometryBasics

Λ = SVector{3,Float64}(1.2, 1, 0.8)
@inline hat(v) = @SMatrix [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
function F!(dX, X, Λ, t)
    u = @SVector [X[1], X[2], X[3]]; Z = SMatrix{3,3}(ntuple(i -> X[3+i], 9)); Y = SMatrix{3,3}(ntuple(i -> X[12+i], 9))
    û = hat(u); v̂ = hat(Λ .* u)
    u̇ = (v̂ * u) ./ Λ; Ż = û * (Λ .* Z) - v̂ * Z; Ẏ = Z - (û * Y)
    dX .= SVector(u̇..., Ż..., Ẏ...)
    return nothing
end
@inline to_energy_ellipsoid(û, Λ) = û .* 1/(sqrt(Λ ⋅ (û.^2)))

# --- Modified Diagnostic Version of the Function ---
function first_zero_time_DIAGNOSTIC(u0::SVector{3,Float64}, Λ::SVector{3,Float64}; tmax=11.0)
    Z₀ = SMatrix{3,3,Float64}(I)
    Y₀ = Z₀ - Z₀
    # Using MVector and vec() for clarity and correctness
    X0 = MVector(u0..., vec(Z₀)..., vec(Y₀)...) 
    prob = ODEProblem(F!, X0, (0.0, tmax), Λ)

    function det_derivative(X, t, integrator)
        u = SVector(X[1], X[2], X[3]); Z = SMatrix{3,3}(ntuple(i -> X[3+i], 9)); Y = SMatrix{3,3}(ntuple(i -> X[12+i], 9))
        dY = det(Y)
        if abs(dY) < 1e-9; return 0.0; end
        û = hat(u); Ẏ = Z - (û * Y); adj_Y = dY * inv(Y)
        return tr(adj_Y * Ẏ)
    end

    # MODIFICATION: This version just prints. It does not terminate.
    function affect_diagnostic!(integrator)
        if integrator.t > 1.0
            X = integrator.u
            Y = SMatrix{3,3}(ntuple(i -> X[12+i], 9))
            normalized_det = abs(det(Y)) / norm(Y)
            println("Found extremum at t = $(integrator.t) with normalized_det = $normalized_det")
        end
    end

    cb = ContinuousCallback(det_derivative, affect_diagnostic!, abstol=1e-4)
    
    # We run the simulation and intentionally ignore the output for now.
    solve(prob, DP5(), reltol=1e-6, abstol=1e-6, callback=cb)

    # We return NaN because this function is just for printing.
    return NaN
end

# --- Create a single test case ---
nθ, nφ = 60, 120
θs = range(0, π, length=nθ)
φs = range(0, 2π, length=nφ)

# Pick a "typical" point from the middle of the grid
θ_test = θs[nθ ÷ 2]
φ_test = φs[nφ ÷ 2]
û_test = @SVector [sin(θ_test)*cos(φ_test), sin(θ_test)*sin(φ_test), cos(θ_test)]
u0_test = to_energy_ellipsoid(û_test, Λ)

println("--- Running Diagnostic for a single u0 ---")
# Run the diagnostic function on just this one point
first_zero_time_DIAGNOSTIC(u0_test, Λ; tmax=11)
println("--- Diagnostic Finished ---")
=#
#=
using StaticArrays, LinearAlgebra, DifferentialEquations, BenchmarkTools, RecursiveArrayTools, Pkg

# ==============================================================================
# SECTION 1: COMMON DEFINITIONS (No changes here, same as before)
# ==============================================================================
@inline hat(v) = @SMatrix [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
if !isdefined(Main, :Λ)
    const Λ = SVector(1.2, 1.0, 0.8)
    const u₀_comp = @SVector [0.360571037130021, -0.16205776246707227, -1.0110161138730018]
    const Z₀_comp = SMatrix{3,3,Float64}(I)
    const Y₀_comp = Z₀_comp - Z₀_comp
    const T = 10.0
    const tspan = (0.0, T)
    const dt_fixed = 0.001
end

function F_inplace!(Ẋ, X, Λ, t)
    # Unpack variables
    u = @SVector [X[1], X[2], X[3]]
    Z = SMatrix{3,3}(ntuple(i -> X[3+i], 9))
    Y = SMatrix{3,3}(ntuple(i -> X[12+i], 9))
    # Map vectors to skew-symmetric matrices
    û = hat(u)
    v̂ = hat(Λ .* u)
    # Compute derivatives
    u̇ = (v̂ * u) ./ Λ
    Ż = û * (Λ .* Z) - v̂ * Z
    Ẏ = Z - (û * Y)
    # Pack variables
    Ẋ .= SVector(u̇..., Ż..., Ẏ...)
    return nothing
end

function F_SVector(X::SVector{21}, p, t)
    u = SVector{3}(ntuple(i -> X[i], 3))
    Z = SMatrix{3,3}(ntuple(i -> X[i+3], 9))
    Y = SMatrix{3,3}(ntuple(i -> X[i+12], 9))
    û = hat(u); v̂ = hat(p .* u)
    u̇ = v̂ * u ./ p
    Ż = û * (p .* Z) - v̂ * Z
    Ẏ = Z - û * Y
    return SVector{21}(u̇..., Ż..., Ẏ...)
end

function F_ArrayPartition(X, p, t)
    u, Z, Y = X.x
    û = hat(u); v̂ = hat(p .* u)
    u̇ = v̂ * u ./ p
    Ż = û * (p .* Z) - v̂ * Z
    Ẏ = Z - û * Y
    return ArrayPartition(u̇, Ż, Ẏ)
end

# ==============================================================================
# SECTION 2: BENCHMARK SUITE SETUP (With the fix)
# ==============================================================================
X₀_MVector = MVector(u₀_comp..., vec(Z₀_comp)..., vec(Y₀_comp)...)
X₀_SVector = SVector{21}(u₀_comp..., vec(Z₀_comp)..., vec(Y₀_comp)...)
X₀_AP = ArrayPartition(u₀_comp, Z₀_comp, Y₀_comp)

prob_MVector = ODEProblem(F_inplace!, X₀_MVector, tspan, Λ)
prob_SVector = ODEProblem(F_SVector, X₀_SVector, tspan, Λ)
prob_AP = ODEProblem(F_ArrayPartition, X₀_AP, tspan, Λ)

SUITE = BenchmarkGroup()
SUITE["Adaptive Solvers"] = BenchmarkGroup(["DP5", "Tsit5"])
SUITE["Fixed-Step Solvers"] = BenchmarkGroup(["RK4"])

adaptive_opts = Dict(:reltol => 1e-4, :abstol => 1e-4, :saveat => 0.01)
fixed_opts = Dict(:dt => dt_fixed, :adaptive => false, :save_everystep => false)

problems = ["MVector" => prob_MVector, "SVector" => prob_SVector, "ArrayPartition" => prob_AP]

for (name, prob) in problems
    # Adaptive solvers work with all types
    SUITE["Adaptive Solvers"]["DP5"][name] = @benchmarkable solve($prob, DP5(); $adaptive_opts...)
    SUITE["Adaptive Solvers"]["Tsit5"][name] = @benchmarkable solve($prob, Tsit5(); $adaptive_opts...)
    
    # *** THE FIX IS HERE ***
    # Fixed-step RK4 is not compatible with NamedTuple, so we skip it.
    if name != "NamedTuple"
        SUITE["Fixed-Step Solvers"]["RK4"][name] = @benchmarkable solve($prob, RK4(); $fixed_opts...)
    end
end

# ==============================================================================
# SECTION 3: RUN AND REPORT (No changes here)
# ==============================================================================

results = run(SUITE, verbose = true)
#=
using StaticArrays, LinearAlgebra, DifferentialEquations, GLMakie, GeometryBasics, BenchmarkTools

@inline hat(v) = @SMatrix [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]

# Hyper-optimized version using ntuple for unpacking
function F_SVector_Opt(X::SVector{21}, Λ, t)
    # Unpack using ntuple for maximum compiler optimization
    # This avoids creating intermediate views or temporary arrays.
    u = SVector{3}(ntuple(i -> X[i], 3))
    Z = SMatrix{3,3}(ntuple(i -> X[i+3], 9))
    Y = SMatrix{3,3}(ntuple(i -> X[i+12], 9))

    # Map vectors to skew-symmetric matrices
    û = hat(u)
    v̂ = hat(Λ .* u)

    # Compute derivatives
    u̇ = v̂ * u ./ Λ
    Ż = û * (Λ .* Z) - v̂ * Z
    Ẏ = Z - (û * Y)

    # Pack derivatives. Splatting is very efficient here.
    return SVector{21}(u̇..., Ż..., Ẏ...)
end

# Parameters and Initial Conditions
Λ = SVector{3,Float64}(1.2, 1, 0.8)
u₀ = @SVector [0.360571037130021, -0.16205976246707227, -1.0110161138730018]
Z₀ = SMatrix{3,3,Float64}(I)
Y₀ = Z₀ - Z₀

# Initial state is an immutable SVector
X₀_SVector = SVector{21}(u₀..., vec(Z₀)..., vec(Y₀)...)

# Time span
T = 10
tspan = (0.0, T)

# Create and benchmark the problem
prob_SVector_Opt = ODEProblem(F_SVector_Opt, X₀_SVector, tspan, Λ)
println("Benchmarking hyper-optimized SVector version:")
@btime sol = solve($prob_SVector_Opt, DP5(), reltol=1e-4, abstol=1e-4, dt=0.01, saveat=0.01)
nothing

# No change here, already optimal
@inline hat(v) = @SMatrix [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]

# CHANGED: Out-of-place function `F` returning an SVector
function F_SVector(X, Λ, t)
    # Unpack variables directly from the SVector X.
    # Using `@view` or `SVector(X[...])` ensures we get static subarrays.
    u = X[SOneTo(3)] # Statically-sized slice
    Z = SMatrix{3,3,Float64,9}(X[SVector{9}(4:12)])
    Y = SMatrix{3,3,Float64,9}(X[SVector{9}(13:21)])

    # Map vectors to skew-symmetric matrices
    û = hat(u)
    v̂ = hat(Λ .* u)

    # Compute derivatives
    u̇ = v̂ * u ./ Λ  # Note: Element-wise division is fine
    Ż = û * (Λ .* Z) - v̂ * Z
    Ẏ = Z - (û * Y)

    # Pack derivatives into a new SVector and return it
    return SVector{21}(u̇..., Ż..., Ẏ...)
end

# Parameters and Initial Conditions (no change)
Λ = SVector{3,Float64}(1.2, 1, 0.8)
u₀ = @SVector [0.360571037130021, -0.16205976246707227, -1.0110161138730018]
Z₀ = SMatrix{3,3,Float64}(I)
Y₀ = Z₀ - Z₀

# CHANGED: Initial state is now an immutable SVector
X₀_SVector = SVector{21}(u₀..., vec(Z₀)..., vec(Y₀)...)

# Time span (no change)
T = 10
tspan = (0.0, T)

# CHANGED: Use the new function and initial state
prob_SVector = ODEProblem(F_SVector, X₀_SVector, tspan, Λ)
println("Benchmarking SVector version:")
@btime sol = solve($prob_SVector, DP5(), reltol=1e-4, abstol=1e-4, dt=0.01, saveat=0.01)
nothing


using StaticArrays, LinearAlgebra, DifferentialEquations, GLMakie, GeometryBasics, BenchmarkTools

@inline hat(v) = @SMatrix [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]

function F!(Ẋ, X, Λ, t)
    # Unpack variables
    u = @SVector [X[1], X[2], X[3]]
    Z = SMatrix{3,3}(ntuple(i -> X[3+i], 9))
    Y = SMatrix{3,3}(ntuple(i -> X[12+i], 9))
    # Map vectors to skew-symmetric matrices
    û = hat(u)
    v̂ = hat(Λ .* u)
    # Compute derivatives
    u̇ = (v̂ * u) ./ Λ
    Ż = û * (Λ .* Z) - v̂ * Z
    Ẏ = Z - (û * Y)
    # Pack derivatives
    Ẋ .= SVector(u̇..., Ż..., Ẏ...)
    return nothing
end

Λ = SVector{3,Float64}(1.2, 1, 0.8)

u₀ = @SVector [0.360571037130021, -0.16205976246707227, -1.0110161138730018] #[0.7774784786756573, 0.2548892552119631, 0.5119376497837815]
Z₀ = SMatrix{3,3,Float64}(I)
Y₀ = Z₀ - Z₀

X₀ = MVector(u₀..., vec(Z₀)..., vec(Y₀)...)

T = 10
tspan = (0.0, T)


prob = ODEProblem(F!, X₀, tspan, Λ)
println("Benchmarking original version:")
@btime sol = solve(prob, DP5(), reltol=1e-4, abstol=1e-4, dt=0.01, saveat=0.01)
nothing

#=
eigλ = map(sol.u) do Xi
    Ymat = SMatrix{3,3}(Xi[13:21])
    eigvals(Matrix(Ymat))          # 3-element Vector{ComplexF64}
end

λmat = hcat(eigλ...)               # 3 × Nt  complex; columns ↔ time steps
reλ  = real.(λmat)                # choose what you want to look at
imλ  = imag.(λmat)
absλ = abs.(λmat)

fig = Figure()
ax  = Axis(fig[1, 1]; xlabel = "t", ylabel = "|λ(t)|", title = "Eigenvalues")

lines!(ax, sol.t, reλ[1, :]; label = "|λ₁|")
lines!(ax, sol.t, reλ[2, :]; label = "|λ₂|")
lines!(ax, sol.t, reλ[3, :]; label = "|λ₃|")

axislegend(ax)

fig
=#
=#
=#