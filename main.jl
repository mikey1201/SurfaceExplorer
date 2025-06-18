using StaticArrays, LinearAlgebra, DifferentialEquations, GLMakie, GeometryBasics, BenchmarkTools

@inline hat(v) = @SMatrix [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
function F!(dX, X, Λ, t)
    u = @SVector [X[1], X[2], X[3]]
    Γ = SMatrix{3,3}(ntuple(i -> X[3+i], 9))
    Λu = Λ .* u
    u̇ = cross(Λu, u) ./ Λ
    Γ̇ = Γ * hat(u)
    dX .= SVector(u̇..., Γ̇...)
    return nothing
end
Λ = SVector{3,Float64}(1, 2, 5) # A B C - principal moments of inertia
u₀ = SVector{3,Float64}(1, 1, 1) # initial angular velocity
T = 1                       # end time

Γ₀ = SMatrix{3,3,Float64}(I)
X₀ = MVector(u₀..., vec(Γ₀)...)
tspan = (0.0, T)

@inline to_energy_ellipsoid(û::SVector{3,Float64}, Λ::SVector{3,Float64}) = begin
    E = Λ ⋅ (û.^2)
    u = û .* 1/(sqrt(E))
    return u
end
function first_zero_time(u0::SVector{3,Float64}, Λ::SVector{3,Float64};
                         tmax=60.0, dt=0.01)
    X0 = MVector(u0..., vec(Γ₀)...)
    prob = ODEProblem(F!, X0, (0.0, tmax), Λ)
    sol = solve(prob, DP5(), reltol=1e-4, abstol=1e-4, dt=dt, saveat=dt)

    for (i, t) in enumerate(sol.t)
        if i+1 <= length(sol.t)
            Γ_t1 = SMatrix{3,3}(ntuple(j -> sol.u[i][3+j], 9))
            Γ_t2 = SMatrix{3,3}(ntuple(j -> sol.u[i+1][3+j], 9))
            v1 = Γ_t1 * u0
            v2 = Γ_t2 * u0
            condition1 = dot(v1, cross(u0, Λ .* u0))
            condition2 = dot(v2, cross(u0, Λ .* u0))
            #println(condition)
            if condition1 * condition2 < 0
                #println(t)
                return t
            end
        end
    end
    return NaN
end


nθ, nφ = 200, 200
θs = range(0, π, length=nθ)
φs = range(0, 2π, length=nφ)


Tmat = Matrix{Float64}(undef, nθ, nφ)
x = Matrix{Float64}(undef, nθ, nφ)
y = similar(x)
z = similar(x)
function cartesian_to_spherical(p::NTuple{3,Float64})
    x,y,z = p
    r = sqrt(x^2 + y^2 + z^2)
    θ = acos(clamp(z/r, -1, 1))
    φ = atan(y, x)
    if φ < 0
        φ += 2π 
    end
    return θ, φ, r
end
θarr = Array{Float64}(undef, nθ, nφ)
φarr = similar(θarr)


for i in 1:nθ, j in 1:nφ
    θ = θs[i]
    φ = φs[j]
    û = @SVector [sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)]
    u0 = to_energy_ellipsoid(û, Λ)
    Tval = first_zero_time(u0, Λ; tmax=100, dt=0.01)
    Tmat[i,j] = Tval

    if isnan(Tval)
        x[i,j] = 1; y[i,j] = 1; z[i,j] = 1
        continue
    end

    x[i,j] = Tval * u0[1]
    y[i,j] = Tval * u0[2]
    z[i,j] = Tval * u0[3]

    θ, φ, r = cartesian_to_spherical((x[i,j], y[i,j], z[i,j]))

    θarr[i,j] = θ
    φarr[i,j] = φ
end

# Create solid surface plot
fig = Figure(size=(600,400))
ax = Axis3(fig[1,1], aspect=(1,1,1))

surface!(ax, x, y, z;)

fig



#=
prob = ODEProblem(F!, X₀, tspan, Λ)
sol = solve(prob, DP5(), reltol=1e-4, abstol=1e-4, dt=0.01, saveat=0.01)

function LI!(LI, sol, u₀, uxω)
    @inbounds @simd for i in eachindex(sol.t)
        X = sol.u[i]
        Γ = SMatrix{3,3}(ntuple(i -> X[3+i], 9))
        v = Γ * u₀
        LI[i] = v ⋅ uxω
    end
    return LI
end
#=What I had in mind was considering only unit vectors, i.e., rescaling so that
Au_1^2 + Bu_2^2 + Cu_3^2 = 1.
Then you’d find the smallest time T>0 where the quantity is zero,
and plot the vector T(u_1,u_2,u_3) as a single point.
Do this over all possible unit vectors and you should get something like a surface. 
That surface is what I’d like to see.=#
LI = Vector{Float64}(undef, length(sol.t))
uxω = SVector{3,Float64}(cross(u₀, Λ .* u₀)) # Λ = A B C = 1 2 5
LI!(LI, sol, u₀, uxω)
=#

#=
function makesurface(samplecount)
    nθ, nφ = 50, 100

    φs = range(0, 2π; length = nφ)
    θs = range(0, π;   length = nθ)

    # build an array of unit‐length SVectors
    u_hat_grid = [ @SVector [sin(θ)*cos(φ),
                             sin(θ)*sin(φ),
                             cos(θ)]
                            for θ in θs, φ in φs ]
    gammas = Vector{}(undef, samplecount)
    for i in eachindex(u)
        X = MVector{12}(u[i]..., vec(Γ₀)...)
        prob = ODEProblem(F!, X, tspan, Λ)
        gammas[i] = solve(prob, DP5(), reltol=1e-4, abstol=1e-4, dt=0.01, saveat=0.04)
    end
    firstzeros = Vector{Float64}(undef, samplecount)
    for i in eachindex(gammas)
        temp = Vector{Float64}(undef, length(gammas[i].t))
        LI!(temp, gammas[i], u[i], cross(u[i], Λ .* u[i]))
        for (k, val) in enumerate(temp)
            if val * temp[1] < 0
                firstzeros[i] = k * 0.01
                break
            end
        end
    end
    #=
    xy = [Point2f(point[1], point[2]) for point in u]  # 2D points for triangulation
    zs = [point[3] for point in u]                 # Z-coordinates
    =#
    xs = [p[1] for p in u]
    ys = [p[2] for p in u]
    zs = [p[3] for p in u]
    # Perform triangulation
    #=
    tri = triangulate([getx.(xy)'; gety.(xy)'])  # Create 2D triangulation
    =#
    PlotlyJS.scatter3d(xs,ys,zs;)
end
makesurface(10001)
=#

#=
Y = Array(sol)
plotlyjs()
plt1 = plot(sol.t, Y[4:12, :]',
    label=["γ₁₁" "γ₁₂" "γ₁₃" "γ₂₁" "γ₂₂" "γ₂₃" "γ₃₁" "γ₃₂" "γ₃₃"],
    xlabel="t", ylabel="y",
    title="Gamma check", size=(800, 800),
    xlims=(0, T), ylims=(-2, 2))
Plots.display(plt1)
=#



#=
println("pre")


println("post")

plt2 = plot(sol.t, LI,
     xlabel="t", ylabel="LI(t)",
     legend=false)
Plots.display(plt1)
=#
#=
v = Γ * u₀
v ⋅ (cross(u₀, Λ .* u₀)) #find when it is 0 (if it is 0) if its never 0 that is of interest too
# what does the time look like as a function of the other variables?
# for each fixed A B C plot the time as a function of the initial velocity
# the initial velocity is a a unit vector, assume conservation law  == 1 au1^2 +....
# for various values of u0 scaled so cons law  = 1 find a time st ... = 0
# get all times and plot them multiplied by the initial velocity
=#


#G(10, )
#plot((0.0, 10), G, [1, 2, 5], [1, 1, 1])

#autostiff 9.76ms 9.4mb autononstiff same
    #=
    plotlyjs()

    Y = Array(sol)
    A, B, C = Λ
    E₀ = Λ ⋅ u₀.^2
#=
    plt = plot(sol.t, A*Y[1,:].^2 + B*Y[2,:].^2 + C*Y[3,:].^2,
    label="u₁² + u₂² + u₃²", xlabel="t", ylabel="value",
    title="Conservation law check", size=(800, 800),
    xlims=(0, T), ylims=(E₀*.5, E₀*1.5))
    =#
    plt = plot(sol.t, Y[4:12, :]',
    label=["γ₁₁" "γ₁₂" "γ₁₃" "γ₂₁" "γ₂₂" "γ₂₃" "γ₃₁" "γ₃₂" "γ₃₃"],
    xlabel="t", ylabel="y",
    title="Gamma check", size=(800, 800),
    xlims=(0, T), ylims=(-2, 2))
    
    Plots.display(plt)
    =#


#@btime sol = solve(prob,alg_hints, reltol=1e-10, abstol=1e-12, saveat=0.01) #vern7 12ms 12mb vern8 8.587ms 7.58mb vern9 8.467ms 7.44mb

# Conservation law (derivative)
#E = 2 * (Λu ⋅ u̇)
#=
return SVector{12}(u̇..., vec(Γ̇)...)           # tell it the length
#    or
@SVector [u̇; vec(Γ̇)]                          # build with a literal
#    or
StaticArrays.SVector{12}(u̇[1],u̇[2],u̇[3], Γ̇.data...)  # explicit
=#



#DP8 < Vern8 < Vern9 < Vern7 < auto=stiff=nonstiff < Tsit5 < Vern6


# Plotting

#=
#### Ellipsoids and Geodesic ####
F₀ = Λ.^2 ⋅ u₀.^2

φ = range(0, 2π; length=120)
θ = range(0, π;  length=60)

ae, be, ce = sqrt.(E₀ ./ Λ)
xE = [ae*sin(θ)*cos(φ) for θ in θ, φ in φ]
yE = [be*sin(θ)*sin(φ) for θ in θ, φ in φ]
zE = [ce*cos(θ)       for θ in θ, φ in φ]

as, bs, cs = sqrt.(F₀ ./ Λ.^2)
xF = [as*sin(θ)*cos(φ) for θ in θ, φ in φ]
yF = [bs*sin(θ)*sin(φ) for θ in θ, φ in φ]
zF = [cs*cos(θ)       for θ in θ, φ in φ]

surface(xE, yE, zE; color = :blue, fillalpha = 1, legend = false)

surface!(xF, yF, zF; color = :red, fillalpha = 0.70)

plot!(sol, idxs = (1, 2, 3), color = :green, lw = 10)
plot!(-sol[1,:], -sol[2,:], -sol[3,:], color = :green, lw = 10,
    xlabel="u₁", ylabel="u₂", zlabel="u₃",
    xlims=(-6,6), ylims=(-6,6), zlims=(-6,6),
    size = (800,800))
#### #### ####
=#
#### Constraint checks ####
#=
# Conservation law check (derivative)
plot(sol.t, sol[13,:], label="E", xlabel="t", ylabel="value",
      title="Conservation law check", size=(800, 800),
      xlims=(0, T), ylims=(E₀*.5, E₀*1.5))
=#
# Conservation law check (direct)


#=
# Orthonormality check
Γ = reshape(view(Y, 4:12, :), 3, 3, :)
γ₁ = @view Γ[1, :, :]
γ₂ = @view Γ[2, :, :]
γ₃ = @view Γ[3, :, :]
dotproducts = [
    vec(sum(γ₁ .* γ₁, dims=1)),
    vec(sum(γ₂ .* γ₂, dims=1)),
    vec(sum(γ₃ .* γ₃, dims=1)),
    vec(sum(γ₁ .* γ₂, dims=1)),
    vec(sum(γ₁ .* γ₃, dims=1)),
    vec(sum(γ₂ .* γ₃, dims=1))
    ]
plot(sol.t, dotproducts, label=["γ₁⋅γ₁" "γ₂⋅γ₂" "γ₃⋅γ₃" "γ₁⋅γ₂" "γ₁⋅γ₃" "γ₂⋅γ₃"],
     xlabel="t", ylabel="value", title="Γ₁ entries",
     size=(1000, 800), xlims=(0, T), ylims=(-1, 2))
#### #### ####
=#

#### Gamma plot ####

#### #### ####
