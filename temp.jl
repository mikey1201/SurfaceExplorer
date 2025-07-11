#=
using StaticArrays, LinearAlgebra, DifferentialEquations, GLMakie, Observables

Z₀ = SMatrix{3,3,Float64}(I)
Y₀ = zeros(9)

@inline hat(v) = @SMatrix [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
@inline adj(X) = @SMatrix [ X[5]*X[9]-X[6]*X[8]  X[3]*X[8]-X[2]*X[9]  X[2]*X[6]-X[3]*X[5];
                            X[6]*X[7]-X[4]*X[9]  X[1]*X[9]-X[3]*X[7]  X[3]*X[4]-X[1]*X[6];
                            X[4]*X[8]-X[5]*X[7]  X[2]*X[7]-X[1]*X[8]  X[1]*X[5]-X[2]*X[4]]
@inline unpackX(X) = SVector{3,Float64}(X[1], X[2], X[3]), SMatrix{3,3}(ntuple(i -> X[3+i], 9)), SMatrix{3,3}(ntuple(i -> X[12+i], 9))

function F(X, Λ, t)
    u, Z, Y = unpackX(X)
    u̇ = (hat(Λ .* u) * u) ./ Λ
    Ż = (hat(u) * (Λ .* Z)) - (hat(Λ .* u) * Z)
    Ẏ = Z - (hat(u) * Y)
    return SVector{21,Float64}(u̇..., Ż..., Ẏ...)
end

function condition(out, X, t, integrator)
    u, Z, Y = unpackX(X)
    Ẏ = Z - (hat(u) * Y)
    out[1] = det(Y)
    out[2] = sum(adj(Y) .* Ẏ)
end

affect_pos!(integ, idx) = terminate!(integ)
affect_neg!(integ, idx) = idx == 1 ? terminate!(integ) : nothing
fzfm = VectorContinuousCallback(condition, affect_pos!, affect_neg!, 2, interp_points=20, save_positions=(false,false))
settings = (callback=fzfm, reltol=1e-5, abstol=1e-5, dense=false, save_everystep=false, save_start=false, save_end=true)

u₀_placeholder = SVector{3,Float64}(1.0, 0.0, 0.0)
X₀_placeholder = SVector{21,Float64}(u₀_placeholder..., Z₀..., Y₀...)
prob_template = ODEProblem(F, X₀_placeholder, (0.0, 15.0), [1.2, 1.0, 0.8])
output_func(sol, i) = (sol.t[end], sol.retcode == :Terminated)

function calculate_surface(Λ_val, nθ, nϕ)
    local_prob = remake(prob_template; p = Λ_val)
    prob_func(p, i, repeat) = remake(p; u0 = SVector(sphere[i]..., Z₀..., Y₀...))
    θ = range(0, π, length = nθ)
    ϕ = range(0, 2π, length = nϕ)
    sphere = vec(SVector.(sin.(θ) .* cos.(ϕ)', sin.(θ) .* sin.(ϕ)', cos.(θ)))
    eprob = EnsembleProblem(local_prob, prob_func=prob_func, output_func=output_func, safetycopy = false)
    sols = solve(eprob, DP5(), EnsembleThreads(); trajectories=nθ*nϕ, settings...)
    Tmat = reshape(first.(sols.u), nθ, nϕ)
    coords = Tmat .* reshape(sphere, nθ, nϕ)
    x, y, z = map(i -> getindex.(coords, i), 1:3)    
    return Tmat, x, y, z
end

RESOLUTIONS = Dict(:Low=>(120,240), :Medium=>(240,480), :High=>(360,720), :Extreme=>(960,1920))
REFINEMENT_ORDER = [:Low, :Medium, :High, :Extreme]

fig = Figure(size=(1400, 900))
ax = Axis3(fig[1:3, 1:3], aspect=(1, 1, 1), title="Conjugacy Surface", xlabel="X", ylabel="Y", zlabel="Z")
target_resolution_obs = Observable((:Low, RESOLUTIONS[:Low]))
displayed_resolution_obs = Observable((:Low, RESOLUTIONS[:Low]))

controls_grid = GridLayout(fig[1:2, 4], tellwidth=true, halign=:left)
Label(controls_grid[1, 1], "Controls", fontsize=20, tellwidth=false)
Label(controls_grid[2, 1], @lift("Target: $(first($target_resolution_obs))"), tellwidth=false)
Label(controls_grid[3, 1], @lift("Displayed: $(first($displayed_resolution_obs))"), tellwidth=false)
sg = SliderGrid(controls_grid[4,1], (label="Λ₁",range=0.01:0.01:5,startvalue=1.2), (label="Λ₂",range=0.01:0.01:5,startvalue=1.0), (label="Λ₃",range=0.01:0.01:5,startvalue=0.8))
Label(controls_grid[5, 1], "Resolution Presets", fontsize=16, tellwidth=false)
button_grid = GridLayout(controls_grid[6, 1], tellwidth=false)
buttons = Dict(name => Button(button_grid[i, 1], label=string(name), width=100) for (i, name) in enumerate(REFINEMENT_ORDER))
Λ_obs = lift(SVector{3,Float64}, sg.sliders[1].value, sg.sliders[2].value, sg.sliders[3].value)

Tmat_obs = Observable(zeros(RESOLUTIONS[:Low]...))
x_obs = Observable(zeros(RESOLUTIONS[:Low]...))
y_obs = Observable(zeros(RESOLUTIONS[:Low]...))
z_obs = Observable(zeros(RESOLUTIONS[:Low]...))
color_range_obs = Observable((0.0, 20.0))
surf = surface!(ax, x_obs, y_obs, z_obs; color=Tmat_obs, colormap=:viridis, colorrange=color_range_obs)
Colorbar(fig[4, 1:3], surf, label="Conjugacy Time (T)", vertical=false, flipaxis=false)
rowgap!(fig.layout, 10); colgap!(fig.layout, 10)

request_channel = Channel{Any}(1)
result_holder = Ref{Any}()
result_trigger = Observable{Int}(0)

on(result_trigger) do _
    if isassigned(result_holder)
        Tmat_new, x_new, y_new, z_new, res_name = result_holder.x
        Tmat_obs[] = Tmat_new
        x_obs[] = x_new
        y_obs[] = y_new
        z_obs[] = z_new
        color_range_obs[] = isempty(Tmat_new) ? (0,1) : extrema(Tmat_new)
        displayed_resolution_obs[] = (res_name, size(Tmat_new))
        println("PLOTTER: Display updated to '$res_name'.")
    end
end

@async begin
    for (Λ_val, res_name, res_tuple) in request_channel
        try
            preview_res_name, preview_res_tuple = REFINEMENT_ORDER[1], RESOLUTIONS[REFINEMENT_ORDER[1]]
            Tmat_new, x_new, y_new, z_new = calculate_surface(Λ_val, preview_res_tuple...)
            result_holder.x = (Tmat_new, x_new, y_new, z_new, preview_res_name)
            result_trigger[] += 1

            target_idx = findfirst(==(res_name), REFINEMENT_ORDER)
            if !isnothing(target_idx) && target_idx > 1
                for i in 2:target_idx
                    if isready(request_channel) break end
                    refine_res_name = REFINEMENT_ORDER[i]
                    refine_res_tuple = RESOLUTIONS[refine_res_name]
                    Tmat_new, x_new, y_new, z_new = calculate_surface(Λ_val, refine_res_tuple...)
                    result_holder.x = (Tmat_new, x_new, y_new, z_new, refine_res_name)
                    result_trigger[] += 1
                end
            end
        catch e
            @error "Error in worker task!" exception=(e, catch_backtrace())
        end
    end
end

function submit_request(Λ_val, res_name, res_tuple)
    while isready(request_channel); take!(request_channel); println("UI: An old request was cancelled."); end
    println("UI: Submitting new request for target '$res_name'.")
    put!(request_channel, (Λ_val, res_name, res_tuple))
end

for (name, res_tuple) in pairs(RESOLUTIONS)
    on(buttons[name].clicks) do _; target_resolution_obs[]=(name,res_tuple); submit_request(Λ_obs[], name, res_tuple); end
end

throttled_Λ_obs = Observables.throttle(0.3, Λ_obs)
on(throttled_Λ_obs) do Λ_val; res_name,res_tuple=target_resolution_obs[]; submit_request(Λ_val, res_name, res_tuple); end

submit_request(Λ_obs[], target_resolution_obs[][1], target_resolution_obs[][2])
limits!(ax, -13, 13, -13, 13, -13, 13)
display(fig)
=#

#=
using StaticArrays, LinearAlgebra, DifferentialEquations, GLMakie, BenchmarkTools, ProgressMeter, DiffEqCallbacks
# --- Core Logic from Original Code ---

# Define constants and utility functions

Z₀ = SMatrix{3,3,Float64}(I)
Y₀ = SMatrix{3,3,Float64}(zeros(9))

@inline hat(v) = @SMatrix [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
@inline unitvector(θ, ϕ) = @SVector [sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ)]
@inline unpackX(X) = SVector{3,Float64}(X[1], X[2], X[3]), SMatrix{3,3}(ntuple(i -> X[3+i], 9)), SMatrix{3,3}(ntuple(i -> X[12+i], 9)) #u, Z, Y

# Differential Equation
function F(X, Λ, t)
    u, Z, Y = unpackX(X)
    u̇ = (hat(Λ .* u) * u) ./ Λ                  # u̇ = ad_u⃰(u)
    Ż = (hat(u) * (Λ .* Z)) - (hat(Λ .* u) * Z) # Ż = ad_u⃰(Z) + ad_Z⃰(u)
    Ẏ = Z - (hat(u) * Y)                        # Ẏ = Z - ad_u(Y)
    return SVector{21,Float64}(u̇..., Ż..., Ẏ...)
end

# Termination condition
condition(out, X, t, integrator) = begin
    u, Z, Y = unpackX(X)
    Ẏ = Z - (hat(u) * Y)
    out[1] = det(Y)
    out[2] = tr(inv(Y) * Ẏ)
end

affect_pos!(integ, idx) = terminate!(integ)
affect_neg!(integ, idx) = idx == 1 ? terminate!(integ) : nothing
fzfm = VectorContinuousCallback(condition, affect_pos!, affect_neg!, 2, interp_points=100)
settings = (callback=fzfm, reltol=1e-4, abstol=1e-4, save_everystep=false, save_start=false, save_end=true)
# --- Main Calculation Function (Now takes resolution as an argument) ---
θ = range(0, π, length=nθ)
ϕ = range(0, 2π, length=nϕ)
u₀ = SVector{3,Float64}(1.0, 0.0, 0.0)
X₀ = SVector{21,Float64}(u₀..., Z₀..., Y₀...)
prob = ODEProblem(F, X₀, (0.0, 15.0), Λ)
output_func(sol, i) = (sol.t[end], sol.retcode == :Terminated)
function calculate_surface(Λ, nθ, nϕ)
    Tmat = Matrix{Float64}(undef, nθ, nϕ)

    function prob_func(prob, i, repeat)
        j = ((i-1) % nθ) + 1
        k = floor(Int, (i-1) / nθ) + 1
        u₀ = sphere[j, k]
        remake(prob; u0=SVector(u₀..., Z₀..., Y₀...))
    end
    eprob = EnsembleProblem(prob, prob_func=prob_func, output_func=output_func, safetycopy=false)   #Ensemble problem seems to be the fastest method

    sphere = [unitvector(θ, ϕ) for θ in θ, ϕ in ϕ]  #Must run before the solve

    sols = solve(eprob, DP5(), EnsembleThreads(); trajectories=nθ*nϕ, settings...)

    Tmat = reshape(first.(sols.u), nθ, nϕ)
    coords = Tmat .* sphere
    x, y, z = map(i -> getindex.(coords, i), 1:3)
    println("\nCalculation finished.")
    return Tmat, x, y, z
end


# --- GLMakie Interactive Setup ---

# Set up the figure and layout
fig = Figure(size=(1400, 900))
ax = Axis3(fig[1:3, 1:3], aspect=(1, 1, 1), title="Conjugacy Surface",
           xlabel="X", ylabel="Y", zlabel="Z")

# --- MODIFIED: Add a Resolution slider to the grid ---
sg = SliderGrid(
    fig[2, 4],
    (label = "Λ₁", range = 0.1:0.1:2.5, startvalue = 1.0),
    (label = "Λ₂", range = 0.1:0.1:2.5, startvalue = 1.0),
    (label = "Λ₃", range = 0.1:0.1:2.5, startvalue = 1.0),
    (label = "Resolution (nθ)", range = 30:10:500, startvalue = 120), # NEW SLIDER
    tellwidth = true,
)

# Create Observables for ALL sliders
Λ1_obs = sg.sliders[1].value
Λ2_obs = sg.sliders[2].value
Λ3_obs = sg.sliders[3].value
resolution_obs = sg.sliders[4].value # NEW RESOLUTION OBSERVABLE

# Combine Λ slider observables into a single SVector observable
Λ_obs = lift(SVector{3,Float64}, Λ1_obs, Λ2_obs, Λ3_obs)

# Perform the initial calculation using the starting values of the sliders
nθ_initial = resolution_obs[]
nϕ_initial = 2 * nθ_initial
Tmat_initial, x_initial, y_initial, z_initial = calculate_surface(Λ_obs[], nθ_initial, nϕ_initial)

# Create Observables for the plot data. Note that their size will change.
Tmat_obs = Observable(Tmat_initial)
x_obs = Observable(x_initial)
y_obs = Observable(y_initial)
z_obs = Observable(z_initial)

# Plot the surface using the data observables
color_range_obs = Observable(extrema(Tmat_initial))
surf = surface!(ax, x_obs, y_obs, z_obs;
                color = Tmat_obs,
                colormap = :viridis,
                colorrange = color_range_obs)

# --- CORRECTED INTERACTIVITY BLOCK ---

# Use `lift` to create a single "trigger" observable that holds a tuple of the latest values.
# It will update whenever EITHER `Λ_obs` or `resolution_obs` changes.
trigger_obs = lift(Λ_obs, resolution_obs) do lambda, resolution
    (lambda, resolution)
end

# Now, `on` listens to the single `trigger_obs`.
# The argument to the `do` block is the value of `trigger_obs`, which is our tuple.
on(trigger_obs) do (Λ, nθ_val) # Unpack the tuple `(Λ, nθ_val)` here
    nϕ_val = 2 * nθ_val
    
    Tmat_new, x_new, y_new, z_new = calculate_surface(Λ, nθ_val, nϕ_val)
    
    x_obs[] = x_new
    y_obs[] = y_new
    z_obs[] = z_new
    Tmat_obs[] = Tmat_new
    
    color_range_obs[] = extrema(Tmat_new)
    #autolimits!(ax)
end

# Display the figure
display(fig)
=#

#=
using StaticArrays, LinearAlgebra, DifferentialEquations, GLMakie, BenchmarkTools
#Initial conditions
Λ = SVector{3,Float64}(0.3, 1, 0.8)
Z₀ = SMatrix{3,3,Float64}(I)
Y₀ = zeros(9)
#Resolution / building unit sphere  #Is it possible to have a better representation of a sphere that still plots easily with glmakie?
nθ, nϕ = 120, 240   #There are many points near the poles
θ = range(0, π, length=nθ)
ϕ = range(0, 2π; length=nϕ)
Tmat = Matrix{Float64}(undef, nθ, nϕ)
@inline adj(X) = @SMatrix [ X[5]*X[9]-X[6]*X[8] X[3]*X[8]-X[2]*X[9] X[2]*X[6]-X[3]*X[5];
                            X[6]*X[7]-X[4]*X[9] X[1]*X[9]-X[3]*X[7] X[3]*X[4]-X[1]*X[6];
                            X[4]*X[8]-X[5]*X[7] X[2]*X[7]-X[1]*X[8] X[1]*X[5]-X[2]*X[4]]                                                #f : X -> adjugate(X)
@inline hat(v) = @SMatrix [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]                                                                    #f : v -> v̂
@inline unpackX(X) = SVector{3,Float64}(X[1], X[2], X[3]), SMatrix{3,3}(ntuple(i -> X[3+i], 9)), SMatrix{3,3}(ntuple(i -> X[12+i], 9))  #u, Z, Y ∈ X
#Differential Equation
function F(X, Λ, t) #Benchmark Results: Out-of-Place (SVector) => Trial(6.039 ms), In-Place (MVector) => Trial(7.095 ms)
    u, Z, Y = unpackX(X)
    u̇ = (hat(Λ .* u) * u) ./ Λ                  #u̇ = ad_u⃰(u)
    Ż = (hat(u) * (Λ .* Z)) - (hat(Λ .* u) * Z) #Ż = ad_u⃰(Z) + ad_Z⃰(u)
    Ẏ = Z - (hat(u) * Y)                        #Ẏ = Z - ad_u(Y)
    return SVector{21,Float64}(u̇..., Ż..., Ẏ...)
end 

condition(out, X, t, integrator) = begin    #The callback will terminate the solver when either of the out variables are equal to 0
    u, Z, Y = unpackX(X)
    Ẏ = Z - (hat(u) * Y)
    out[1] = det(Y)
    out[2] = sum(adj(Y) .* Ẏ)   #d/dt(det(Y)) == 0  here we solve for it by taking the Frobenius inner product of adj(Y) and Ẏ
end                             #this computation involves near singular matrices and forces us to use tighter tolerances / more interpolants to obtain accurate results
affect_pos!(integ, idx) = terminate!(integ)                                                                                 #High interp points allows us to lower the tolerances of the solver
affect_neg!(integ, idx) = idx == 1 ? terminate!(integ) : nothing                                                            #this is desirable because interp points are computationally cheaper than precision
fzfm = VectorContinuousCallback(condition, affect_pos!, affect_neg!, 2, interp_points=20, save_positions=(false,false))     #The combination of using the DP5() solver, interp_points=20, tol=1e-5, and
settings = (callback=fzfm, reltol=1e-5, abstol=1e-5, dense=false, save_everystep=false, save_start=false, save_end=true)    #using a VectorContinuousCallback is a sweet spot for speed and accuracy
#A CallbackSet is actually faster here, but for some reason it's less accurate than the VectorContinuousCallback with these settings
u₀ = SVector{3,Float64}(1.0, 0.0, 0.0)
X₀ = SVector{21,Float64}(u₀..., Z₀..., Y₀...)
prob = ODEProblem(F, X₀, (0.0, 20.0), Λ)

output_func(sol, i) = (sol.t[end], sol.retcode == :Terminated)
function start()
function prob_func(prob, i, repeat)
    j = ((i-1) % nθ) + 1
    k = floor(Int, (i-1) / nθ) + 1
    remake(prob; u0=SVector(sphere[j, k]..., Z₀..., Y₀...))
end

eprob = EnsembleProblem(prob, prob_func=prob_func, output_func=output_func, safetycopy=false)   #Ensemble problem seems to be the fastest method
#Build nθ x nϕ matrix of points (x,y,z) that lie on the unit sphere
sphere = SVector.(sin.(θ) .* cos.(ϕ)', sin.(θ) .* sin.(ϕ)', cos.(θ))
#Run solver with each point as an initial condition, record the time that the conjugacy condition is met
sols = solve(eprob, DP5(), EnsembleThreads(); trajectories=nθ*nϕ, settings...)
#Transform times into a nθ x nϕ matrix
Tmat = reshape(first.(sols.u), nθ, nϕ)
#Multiply each point in the sphere matrix by the corresponding point in the time matrix
coords = Tmat .* sphere

x, y, z = map(i -> getindex.(coords, i), 1:3)
#Plot
fig = Figure(size=(1000,800))
ax = Axis3(fig[1,1], aspect=(1,1,1))
surface!(ax, x, y, z; color=Tmat, colormap=:viridis)
fig


end
start()#@benchmark 
=#


using StaticArrays, LinearAlgebra, DifferentialEquations, GLMakie, BenchmarkTools

# Initial conditions
Λ = SVector{3,Float64}(0.3, 1, 0.8)
Z₀ = SMatrix{3,3,Float64}(I)
Y₀ = @SVector zeros(9)

# Resolution / building unit sphere
nθ, nϕ = 120, 240
θ = range(0, π, length=nθ)
ϕ = range(0, 2π; length=nϕ)

# Build nθ x nϕ matrix of points (x,y,z) that lie on the unit sphere
# This is now a global constant to be accessible by output_func
sphere = SVector.(sin.(θ) .* cos.(ϕ)', sin.(θ) .* sin.(ϕ)', cos.(θ))

@inline adj(X) = @SMatrix [ X[5]*X[9]-X[6]*X[8] X[3]*X[8]-X[2]*X[9] X[2]*X[6]-X[3]*X[5];
                            X[6]*X[7]-X[4]*X[9] X[1]*X[9]-X[3]*X[7] X[3]*X[4]-X[1]*X[6];
                            X[4]*X[8]-X[5]*X[7] X[2]*X[7]-X[1]*X[8] X[1]*X[5]-X[2]*X[4]]
@inline hat(v) = @SMatrix [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
@inline unpackX(X) = SVector{3,Float64}(X[1], X[2], X[3]), SMatrix{3,3}(ntuple(i -> X[3+i], 9)), SMatrix{3,3}(ntuple(i -> X[12+i], 9))

# Differential Equation
function F(X, Λ, t)
    u, Z, Y = unpackX(X)
    u̇ = (hat(Λ .* u) * u) ./ Λ
    Ż = (hat(u) * (Λ .* Z)) - (hat(Λ .* u) * Z)
    Ẏ = Z - (hat(u) * Y)
    return SVector{21,Float64}(u̇..., Ż..., Ẏ...)
end

condition(out, X, t, integrator) = begin
    u, Z, Y = unpackX(X)
    Ẏ = Z - (hat(u) * Y)
    out[1] = det(Y)
    out[2] = sum(adj(Y) .* Ẏ)
end

affect_pos!(integ, idx) = terminate!(integ)
affect_neg!(integ, idx) = idx == 1 ? terminate!(integ) : nothing

function start()
    Tmat = Matrix{Float64}(undef, nθ, nϕ)
    # The rest of the code remains the same for plotting the results
    for I in CartesianIndices(Tmat)
        fzfm = VectorContinuousCallback(condition, affect_pos!, affect_neg!, 2, interp_points=20, save_positions=(false,false))
        settings = (callback=fzfm, reltol=1e-5, abstol=1e-5, dense=false, save_everystep=false, save_start=false, save_end=true)
        i, j = I.I
        u₀ = sphere[i,j]
        X₀ = SVector{21,Float64}(u₀..., Z₀..., Y₀...)
        prob = ODEProblem(F, X₀, (0.0, 15.0), Λ)
        Tval = solve(prob, DP5(); settings...).t[end]
        Tmat[i,j] = Tval
    end
    coords = Tmat .* sphere
    x, y, z = map(i -> getindex.(coords, i), 1:3)

    fig = Figure(size=(1000,800))
    ax = Axis3(fig[1,1], aspect=(1,1,1))
    surface!(ax, x, y, z; color=Tmat, colormap=:viridis)
    
    # Return the figure so it displays
    return fig
end

# To run the simulation and see the output
start()



#=========================================================================================================#
#=
#In-Place Version
function F!(dX, X, Λ, t)
    u, Z, Y = unpackX(X)
    u̇ = (hat(Λ .* u) * u) ./ Λ
    Ż = (hat(u) * (Λ .* Z)) - (hat(Λ .* u) * Z)
    Ẏ = Z - (hat(u) * Y)
    dX .= SVector{21,Float64}(u̇..., Ż..., Ẏ...)
    return nothing
end
function prob_func(prob, i, repeat)
    j = ((i-1) % nθ) + 1
    k = floor(Int, (i-1) / nθ) + 1
    u₀ = sphere[j, k]
    remake(prob; u0=MVector(u₀..., Z₀..., Y₀...))
end
=#
#=
#CallbackSet Version #Doesn't work with sweet spot settings
firstzero(X, t, integrator) = det(unpackX(X)[3])
firstminimum(X, t, integrator) = begin
    u, Z, Y = unpackX(X)
    Ẏ = Z - (hat(u) * Y)
    sum(adj(Y) .* Ẏ)
end
fz = ContinuousCallback(firstzero, terminate!; interp_points=30, save_positions=(false,false))
fm = ContinuousCallback(firstminimum, terminate!, nothing; save_positions=(false,false))
fzfm = CallbackSet(fz, fm)
settings = (callback=fzfm, reltol=1e-5, abstol=1e-5, dense=false, save_everystep=false, save_start=false, save_end=true)
=#
#=
#Threaded loop version
x = similar(Tmat)
y = similar(Tmat)
z = similar(Tmat)
Threads.@threads for I in CartesianIndices(Tmat)
    i, j = I.I
    u₀ = unitvector(θ[i], ϕ[j])
    X₀ = SVector{21,Float64}(u₀..., Z₀..., Y₀...)
    prob = ODEProblem(F, X₀, (0.0, 15.0), Λ)
    Tval = solve(prob, DP5(); settings...).t[end]
    Tmat[i,j] = Tval
    x[i,j], y[i,j], z[i,j] = Tval .* u₀
end
=#
#=========================================================================================================#