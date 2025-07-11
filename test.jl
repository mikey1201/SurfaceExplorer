    #=
    Y₁₁, Y₁₂, Y₁₃, Y₂₁, Y₂₂, Y₂₃, Y₃₁, Y₃₂, Y₃₃ = Y
    Ẏ₁₁ = Z₁₁ - (Y₁₃*u₂ - Y₁₂*u₃)
    Ẏ₁₂ = Z₁₂ - (Y₁₁*u₃ - Y₁₃*u₁)
    Ẏ₁₃ = Z₁₃ - (Y₁₂*u₁ - Y₁₁*u₂)
    Ẏ₂₁ = Z₂₁ - (Y₂₃*u₂ - Y₂₂*u₃)
    Ẏ₂₂ = Z₂₂ - (Y₂₁*u₃ - Y₂₃*u₁)
    Ẏ₂₃ = Z₂₃ - (Y₂₂*u₁ - Y₂₁*u₂)
    Ẏ₃₁ = Z₃₁ - (Y₃₃*u₂ - Y₃₂*u₃)
    Ẏ₃₂ = Z₃₂ - (Y₃₁*u₃ - Y₃₃*u₁)
    Ẏ₃₃ = Z₃₃ - (Y₃₂*u₁ - Y₃₁*u₂)
    Ẏ = Ẏ₁₁, Ẏ₁₂, Ẏ₁₃, Ẏ₂₁, Ẏ₂₂, Ẏ₂₃, Ẏ₃₁, Ẏ₃₂, Ẏ₃₃
    =#
    #=
    Z₁₁, Z₁₂, Z₁₃, Z₂₁, Z₂₂, Z₂₃, Z₃₁, Z₃₂, Z₃₃ = Z
    Ż₁₁ = (λ₃ - λ₂)*(Z₁₃*u₂ + Z₁₂*u₃)
    Ż₁₂ = (λ₁ - λ₃)*(Z₁₃*u₁ + Z₁₁*u₃)
    Ż₁₃ = (λ₂ - λ₁)*(Z₁₂*u₁ + Z₁₁*u₂)
    Ż₂₁ = (λ₃ - λ₂)*(Z₂₃*u₂ + Z₂₂*u₃)
    Ż₂₂ = (λ₁ - λ₃)*(Z₂₃*u₁ + Z₂₁*u₃)
    Ż₂₃ = (λ₂ - λ₁)*(Z₂₂*u₁ + Z₂₁*u₂)
    Ż₃₁ = (λ₃ - λ₂)*(Z₃₃*u₂ + Z₃₂*u₃)
    Ż₃₂ = (λ₁ - λ₃)*(Z₃₃*u₁ + Z₃₁*u₃)
    Ż₃₃ = (λ₂ - λ₁)*(Z₃₂*u₁ + Z₃₁*u₂)
    Ż = Ż₁₁, Ż₁₂, Ż₁₃, Ż₂₁, Ż₂₂, Ż₂₃, Ż₃₁, Ż₃₂, Ż₃₃ 
    =#



#=
Γ_stack = [SMatrix{3,3}(s[4:12]...) for s in sol.u]   # cache rotations
ntime = length(sol.t)

plt = plot(sol.t, Y[13:13, :]', label="", xlabel="t", ylabel="y",
      title="Conservation law check", size=(1200, 800))
Plots.display(plt)
plt = plot(sol.t, Y[4:12, :]',
     label=["γ₁₁" "γ₁₂" "γ₁₃" "γ₂₁" "γ₂₂" "γ₂₃" "γ₃₁" "γ₃₂" "γ₃₃"],
     xlabel="t", ylabel="value", title="Γ entries", size=(1600, 900))
Plots.display(plt)
plt = plot(sol.t, Y[14:19, :]', label="Gamma check", xlabel="t", ylabel="y",
      title="Gamma check", size=(1200, 800))
=#
#=
A, B, C = Λ

E = @. A*sol[1,:]^2 + B*sol[2,:]^2 + C*sol[3,:]^2 # - sol[13,:]

plt = plot(sol.t, E, label="E",
      xlabel="t", ylabel="value", 
      title="Energy conservation check", size=(1200, 800),
      xlims=(0, T), ylims=(0, 15))

Y = Array(sol)
plt = plot(sol.t, Y[[1,2,3], :]',
     label=["u₁" "u₂" "u₃"],
     xlabel="t", ylabel="value",
     size=(1200, 800))

plt = plot(sol, vars=(1, 2, 3),
     label=["u₁" "u₂" "u₃"],
     xlabel="t", ylabel="value",
     size=(1200, 800))
Plots.display(plt)

display(plt)

#plt = plot(sol, vars=(1, 2, 3), label=[], xlabel="u₁(t)", ylabel="u₂(t)", zlabel="u₃(t)")
=#


#=
I3 = SMatrix{3,3}(I)

# Measure the deviation from identity
Γ_errors = [norm(Γ - I3) for Γ in Γ_stack]

# Find first time (after t = 0) where Γ is very close to identity
tol = 1e-5
for (i, err) in enumerate(Γ_errors)
    if i > 1 && err < tol
        println("Γ(t) ≈ I at t ≈ ", sol.t[i])
        break
    end
end
=#

#=
Y = Array(sol)

plt = plot(sol.t, Y[4:12, :]',
     label=["γ₁₁" "γ₁₂" "γ₁₃" "γ₂₁" "γ₂₂" "γ₂₃" "γ₃₁" "γ₃₂" "γ₃₃"],
     xlabel="t", ylabel="value", title="Γ rows", size=(1600, 900))
Plots.display(plt)
=#
#=
anim = @animate for (ti, Γ) in zip(sol.t, Γ_stack)
    Plots.heatmap(Γ; c=:coolwarm, zlims=(-1,1), clims=(-1,1),
            xticks=(1:3, ["x" "y" "z"]),
            yticks=(3:1, ["γ₃" "γ₂" "γ₁"]),
            title="Γ(t)  at t = $(round(ti,digits=2))",
            size=(600, 600),
            speed=2)
end
gif(anim, "30.gif", fps=30)
=#


#=
Y = Array(sol)
plt = plot(sol.t, Y[14:19, :]', label="Gamma check", xlabel="t", ylabel="y",
      title="Gamma check", size=(1200, 800))
Plots.display(plt)
Y = Array(sol)
plt = plot(sol.t, Y[[1,2,3], :]',
     label=["u₁" "u₂" "u₃"],
     xlabel="t", ylabel="value")
Plots.display(plt)
Y = Array(sol)
plt = plot(sol.t, Y[13:13, :]', label="", xlabel="t", ylabel="y",
      title="Conservation law check", size=(1200, 800))
Plots.display(plt)
Y = Array(sol)
plt = plot(sol.t, Y[14:19, :]', label="Gamma check", xlabel="t", ylabel="y",
      title="Gamma check", size=(1200, 800))
Plots.display(plt)
plot!(sol.t, Y[4:12, :]',
     label=["γ₁₁" "γ₁₂" "γ₁₃" "γ₂₁" "γ₂₂" "γ₂₃" "γ₃₁" "γ₃₂" "γ₃₃"],
     xlabel="t", ylabel="value", title="Γ rows", size=(1600, 900))
Plots.display(plt)
=#
#=
# ───────────────────── 3. geometry: a unit cube ──────────────────────────────
body_verts = [SVector{3}(x,y,z)
              for x in (-0.5,0.5), y in (-0.5,0.5), z in (-0.5,0.5)]
faces = [[1,2,4,3], [5,6,8,7], [1,2,6,5],
         [3,4,8,7], [1,3,7,5], [2,4,8,6]]

# helper: rotate SVector vertex → Point3f0
rotated_vertices(Γ) =
    [Point3f0( (Γ * v)... ) for v in body_verts]

# ───────────────────── 4. interactive Makie scene ────────────────────────────
fig  = Figure(resolution = (800,600))
ax   = Axis3(fig[1,1]; aspect = :data, perspectiveness = 0.7,
             xlabel = "x", ylabel = "y", zlabel = "z")

idx  = Observable(1)                       # current time index (1…ntime)

# reactive mesh: updates automatically when `idx[]` changes
mesh!(ax, lift(idx) do k
    verts = rotated_vertices(Γ_stack[k])
    (verts, faces)                         # GLMakie accepts (vertices, faces)
end; color = :orange)

# slider to scrub through frames
slider = Slider(fig[2,1];
                range = 1:ntime,
                startvalue = 1,
                width = Relative(0.9))
connect!(slider.value, idx)                # bind slider → index

fig                     
=#





#=
# ──────────────────────── diagnostics & visualisation ────────────────────────
# helper to grab Γ(t) as 3×3 static matrix
Γ_at(i) = SMatrix{3,3}(sol.u[i][4:12]...)
Γ_stack = [Γ_at(i) for i in eachindex(sol)]   # Vector of 3×3 matrices
println("good")
=#
#=
# 1. Body-rates plot ----------------------------------------------------------
Y = Array(sol)
plt = plot(sol.t, Y[1:3, :]',
     label=["u₁" "u₂" "u₃"],
     xlabel="t", ylabel="value")
Plots.display(plt)
println("good")
plot!(sol.t, Y[13, :], 
    label="u₁² + u₂² + u₃²",   
    xlabel="t", ylabel="value")
println("good")
Plots.display(plt)
=#
#=
ntime   = length(sol.t)
Γ_flat  = Matrix{Float64}(undef, ntime, 9)   # time along the rows
for (k, state) in enumerate(sol.u)
    Γ_flat[k, :] .= vec(SMatrix{3,3}(state[4:12]...))
end
=#
#Γ_stack = [SMatrix{3,3}(Γ_flat[k, :]) for k in 1:ntime]   # <– always works

#=
plt = plot(sol.t, Γ_flat;    # no transpose needed
    lw=1.6,
    label=["γ₁₁" "γ₁₂" "γ₁₃" "γ₂₁" "γ₂₂" "γ₂₃" "γ₃₁" "γ₃₂" "γ₃₃"],  
    legend=:right,
    xlabel="t", ylabel="value",
    title="Orientation-matrix entries")
=#
#=
plt = plot(sol.t, Y[4:12, :]',
     label=["γ₁₁" "γ₁₂" "γ₁₃" "γ₂₁" "γ₂₂" "γ₂₃" "γ₃₁" "γ₃₂" "γ₃₃"],
     xlabel="t", ylabel="value", title="Orientation-matrix entries", size=(1200, 800))
Plots.display(plt)
=#

# 4. Orthogonality error ------------------------------------------------------
#Γ_err = [maximum(abs.(Γ*Γ' - I)) for Γ in Γ_stack]
#println("max ‖ΓᵀΓ − I‖∞ ≈ ", maximum(Γ_err))
#println("good")


#=
# 5. (Optional) animated heat-map of Γ(t)  ------------------------------------
# uncomment if you’d like an animation  ─ requires ffmpeg or ImageMagick

=#







#plot(sol.t, sol[13,:], label="E", xlabel="t", ylabel="value", title="Energy")


#plot!(sol.t, sol[1,:], label="u₁", xlabel="t", ylabel="value", title="Body rates")
#plot!(sol.t, sol[2,:], label="u₂")
#plot!(sol.t, sol[3,:], label="u₃")

    #dX[13] = 2*(A*u₁*dX[1] + B*u₂*dX[2] + C*u₃*dX[3])

#=
    #u₁', u₂', u₃'
    dX[1] = (B-C)/A * u₂*u₃
    dX[2] = (C-A)/B * u₁*u₃
    dX[3] = (A-B)/C * u₁*u₂ 
    =#


    #=
    Non matrix form
    dX[4]  = u₃*γ₁₂ - u₂*γ₁₃
    dX[5]  = u₁*γ₁₃ - u₃*γ₁₁
    dX[6]  = u₂*γ₁₁ - u₁*γ₁₂
    dX[7]  = u₃*γ₂₂ - u₂*γ₂₃
    dX[8]  = u₁*γ₂₃ - u₃*γ₂₁
    dX[9]  = u₂*γ₂₁ - u₁*γ₂₂
    dX[10] = u₃*γ₃₂ - u₂*γ₃₃
    dX[11] = u₁*γ₃₃ - u₃*γ₃₁    
    dX[12] = u₂*γ₃₁ - u₁*γ₃₂
    =#
  #=
    dX[14:19] = @SVector [  
        (γ₁₁^2 + γ₁₂^2 + γ₁₃^2),
        (γ₂₁^2 + γ₂₂^2 + γ₂₃^2),
        (γ₃₁^2 + γ₃₂^2 + γ₃₃^2),
        (γ₁₁*γ₂₁ + γ₁₂*γ₂₂ + γ₁₃*γ₂₃),
        (γ₁₁*γ₃₁ + γ₁₂*γ₃₂ + γ₁₃*γ₃₃),   
        (γ₂₁*γ₃₁ + γ₂₂*γ₃₂ + γ₂₃*γ₃₃)]
    =#
    #=
, dotproducts...

    #Orthonormality check
    γ₁, γ₂, γ₃ = eachrow(Γ)
    dotproducts = γ₁⋅γ₁, γ₂⋅γ₂, γ₃⋅γ₃, γ₁⋅γ₂, γ₁⋅γ₃, γ₂⋅γ₃
    dotproducts₀ = 1, 1, 1, 0, 0, 0
    , dotproducts₀...
    =#