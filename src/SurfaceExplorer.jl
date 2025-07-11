using GLMakie, StaticArrays, LinearAlgebra, DifferentialEquations, Observables
#Initial conditions
const u₀ = SVector{3,Float64}(1.0, 0.0, 0.0)
const Z₀ = SMatrix{3,3,Float64}(I)
const Y₀ = @SVector zeros(9)
const X₀ = SVector{21,Float64}(u₀..., Z₀..., Y₀...)
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
#Computing the geodesic u̇ (Euler-Arnold), the linearized velocity Ż (Linearized Euler-Arnold), and the Jacobi field Ẏ
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
affect_neg!(integ, idx) = idx == 1 ? terminate!(integ) : nothing                                                                #this is desirable because interp points are computationally cheaper than precision
const fzfm = VectorContinuousCallback(condition, affect_pos!, affect_neg!, 2, interp_points=20, save_positions=(false,false))   #The combination of using the DP5() solver, interp_points=20, tol=1e-5, and
const settings = (callback=fzfm, reltol=1e-5, abstol=1e-5, dense=false, save_everystep=false, save_start=false, save_end=true)  #a VectorContinuousCallback is a sweet spot for speed and accuracy
#A CallbackSet is faster, but for some reason it's less accurate than the VectorContinuousCallback with these settings
const prob₀ = ODEProblem(F, X₀, (0.0, 15.0), [1.2, 1.0, 0.8])

#========================================================================================================#

output_func(sol, i) = (sol.t[end], sol.retcode == :Terminated)

function createsurface(Λ, res)
    nθ, nϕ, sphere = resdict[res]..., spheres[res]
    prob = remake(prob₀; p=Λ)
    prob_func(p, i, repeat) = remake(p; u0=SVector(sphere[i]..., Z₀..., Y₀...))
    ensembleprob = EnsembleProblem(prob, prob_func=prob_func, output_func=output_func, safetycopy=false)
    sols = solve(ensembleprob, DP5(), EnsembleSplitThreads(); trajectories=nθ*nϕ, settings...)
    Tmat = reshape(first.(sols.u), nθ, nϕ)
    coords = Tmat .* reshape(sphere, nθ, nϕ)
    x, y, z = @SVector [getindex.(coords, i) for i in 1:3]
    return Tmat, x, y, z
end

#========================================================================================================#

createsphere(nθ, nϕ) = vec([SVector(sin(i)*cos(j), sin(i)*sin(j), cos(i)) for i in range(0, π, nθ), j in range(0, 2π, nϕ)])

const resolutions = [:Low, :Medium, :High, :Extreme]
const resdict = Dict(resolutions .=> ((120,240), (240,480), (360,720), (960,1920)))
const spheres = Dict(name => createsphere(dims...) for (name, dims) in resdict)

const reqchannel = Channel{Any}(1)
const resholder = Ref{Any}()
const restrigger = Observable{Int}(0)

function changeres!(Λ, targetres)
    println("WORKER: Changing resolution to '$targetres'.")
    isready(reqchannel) && return println("WORKER: New request cancelled resolution change. Stopping.")
    Tmat, x, y, z = createsurface(Λ, targetres)
    isready(reqchannel) && return println("WORKER: New request arrived resolution change. Discarding result.")
    resholder.x = Tmat, x, y, z, targetres
    notify(restrigger)
end

function submitrequest(Λ, res_name, res_tuple)
    while isready(reqchannel); take!(reqchannel); println("UI: An old request was cancelled."); end
    println("UI: Submitting new request for target '$res_name'.")
    put!(reqchannel, (Λ, res_name, res_tuple))
end

function initgui()
    fig = Figure(size=(1400, 900))  #First we make the window and all of the gui elements
    cg = GridLayout(fig[1:4,4])
    Box(cg[1:5,1:6], color=RGBf(0.95, 0.95, 0.95), strokecolor=:lightgrey, cornerradius=20, alignmode=Outside(-10))
    Label(cg[1,1:6], "Controls", fontsize=22, tellwidth=false)
    Label(cg[2,1:6], "Lambdas", fontsize=16, tellwidth=false)
    sliders = SliderGrid(cg[3,2:5], [(label=i, range=0.01:0.01:5, startvalue=1.0, color_inactive=:lightgrey) for i in ["Λ₁","Λ₂","Λ₃"]]...)   #Sliders to control Λ values
    Label(cg[4,1:6], "Resolution", fontsize=16, tellwidth=false)
    menu = Menu(cg[5,3:4], options=resolutions, selection_cell_color_inactive=:lightgrey)    #Basic resolutions menu
    return fig, sliders, menu
end

function initobservables(fig, sliders)
    return (lift(SVector{3,Float64}, [s.value for s in sliders.sliders]...),
            Observable((:Low, resdict[:Low])),
            Axis3(fig[1:4, 1:3], aspect=(1, 1, 1), xlabel="X", ylabel="Y", zlabel="Z", limits=(-13, 13, -13, 13, -13, 13)),
            [Observable(zeros(resdict[:Low]...)) for _ in 1:4]...,
            Observable((0.0, 20.0)))
end

function start()
    fig, sliders, menu = initgui()
    Λ, targetres, ax, Tmat, x, y, z, colorrange = initobservables(fig, sliders)
    #Bind user input to actions
    on(s -> (targetres[] = (s, resdict[s]); submitrequest(Λ[], s, resdict[s])) , menu.selection)  #on menu selection submit request
    on(sliderchange -> submitrequest(sliderchange, targetres[]...), throttle(0.01, Λ)) #on slider move submit request; throttle to reduce unused computations
    on(_ -> if isassigned(resholder) Tmat[], x[], y[], z[], res = resholder.x; colorrange[] = extrema(Tmat[]) end, restrigger)  #on request submitted update variables

    submitrequest(Λ[], targetres[]...) #First request
    surface!(ax, x, y, z; color=Tmat, colormap=:viridis, colorrange=colorrange) #Plot first surface
    display(fig)    #Display everything to the window
    #
    @async for (Λ, targetres) in reqchannel   #When a request is a made compute surface in low resolution, then the target resolution
        try
            changeres!(Λ, :Low)
            targetres != :Low && changeres!(Λ, targetres)
        catch e
            @error "Error in worker task!" exception=(e, catch_backtrace())
        end
    end
end

#========================================================================================================#

start()