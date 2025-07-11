using GLMakie, StaticArrays, LinearAlgebra, DifferentialEquations, Observables
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
const u₀ = SVector{3,Float64}(1.0, 0.0, 0.0)
const X₀ = SVector{21,Float64}(u₀..., Z₀..., Y₀...)
const prob₀ = ODEProblem(F, X₀, (0.0, 15.0), [1.2, 1.0, 0.8])

#========================================================================================================#

const resolutions = [:Low, :Medium, :High, :Extreme]
const resdict = Dict(resolutions .=> ((120,240), (240,480), (360,720), (960,1920)))
const reqchannel = Channel{Any}(1)
const resholder = Ref{Any}()
const restrigger = Observable{Int}(0)

function do_refinement!(Λ, targetres)
    println("WORKER: Refining to '$targetres'.")
    isready(reqchannel) && return println("WORKER: New request cancelled refinement. Stopping.")
    Tmat, x, y, z = createsurface(Λ, targetres)
    isready(reqchannel) && return println("WORKER: New request arrived during refinement. Discarding result.")
    resholder.x = Tmat, x, y, z, targetres
    notify(restrigger)
end

function submit_request(Λ, res_name, res_tuple)
    while isready(reqchannel); take!(reqchannel); println("UI: An old request was cancelled."); end
    println("UI: Submitting new request for target '$res_name'.")
    put!(reqchannel, (Λ, res_name, res_tuple))
end

function start()
    fig = Figure(size=(1400, 900))  #First we make the window and all of the gui elements
    ax = Axis3(fig[1:3, 1:3], aspect=(1, 1, 1), title="Conjugacy Surface", xlabel="X", ylabel="Y", zlabel="Z", limits=(-13, 13, -13, 13, -13, 13))
    targetres, displayedres = [Observable((:Low, resdict[:Low])) for i in 1:2]   #Observables let us hotswap variables
    box = Box(fig[1:4, 4], color=:white, strokecolor=RGBf(0.8, 0.8, 0.8), cornerradius=20)
    cg = GridLayout(fig[1:2, 4], tellwidth=true, halign=:left)
    Label(cg[1, 1], "Controls", fontsize=20, tellwidth=false)
    sg = SliderGrid(cg[2,1], [(label=i, range=0.01:0.01:5, startvalue=1.0) for i in ["Λ₁","Λ₂","Λ₃"]]...)   #Sliders to control Λ values
    Λ = lift(SVector{3,Float64}, [sg.sliders[i].value for i in 1:3]...)
    Label(cg[3, 1], "Resolution Presets", fontsize=16, tellwidth=false)
    Label(cg[4, 1], @lift("Target: $(first($targetres))"), tellwidth=false)
    Label(cg[5, 1], @lift("Displayed: $(first($displayedres))"), tellwidth=false)
    menu = Menu(cg[6,1], options=resolutions)   #Basic resolutions menu
    Tmat, x, y, z = [Observable(zeros(resdict[:Low]...)) for i in 1:4]
    colorrange = Observable((0.0, 20.0))
    surf = surface!(ax, x, y, z; color=Tmat, colormap=:viridis, colorrange=colorrange)
    Colorbar(fig[4, 1:3], surf, label="Conjugacy Time (T)", vertical=false, flipaxis=false)
    on(menu.selection) do s; targetres[] = (s, resdict[s]); submit_request(Λ[], s, resdict[s]); end #Bind user inputs to actions
    on(sliderchange -> submit_request(sliderchange, targetres[]...), throttle(0.01, Λ)) #Throttle the sliders for smoother experience
    submit_request(Λ[], targetres[]...) #First surface
    display(fig)    #Display everything to the window

    on(restrigger) do _ #When the resolution is changed we need to update all of our variables
        if isassigned(resholder)
            Tmat[], x[], y[], z[], res = resholder.x
            colorrange[] = isempty(Tmat[]) ? (0,1) : extrema(Tmat[])
            displayedres[] = (res, resdict[res])
            println("PLOTTER: Display updated to '$res'.")
        end
    end

    @async for (Λ, targetres) in reqchannel   #When a request is a made compute surface in low resolution, then the target resolution
        try
            do_refinement!(Λ, :Low)
            targetres != :Low && do_refinement!(Λ, targetres)
        catch e
            @error "Error in worker task!" exception=(e, catch_backtrace())
        end
    end
end

#========================================================================================================#

output_func(sol, i) = (sol.t[end], sol.retcode == :Terminated)

createsphere(nθ, nϕ) = vec([SVector(sin(i)*cos(j), sin(i)*sin(j), cos(i)) for i in range(0, π, nθ), j in range(0, 2π, nϕ)])

const spheres = Dict(name => createsphere(dims...) for (name, dims) in resdict)

function createsurface(Λ, res)
    nθ, nϕ, sphere = resdict[res]..., spheres[res]
    prob = remake(prob₀; p=Λ)
    prob_func(p, i, repeat) = remake(p; u0=SVector(sphere[i]..., Z₀..., Y₀...))
    ensembleprob = EnsembleProblem(prob, prob_func=prob_func, output_func=output_func, safetycopy=false)
    sols = solve(ensembleprob, DP5(), EnsembleThreads(); trajectories=nθ*nϕ, settings...)
    Tmat = reshape(first.(sols.u), nθ, nϕ)
    coords = Tmat .* reshape(sphere, nθ, nϕ)
    x, y, z = [getindex.(coords, i) for i in 1:3]
    return Tmat, x, y, z
end

#========================================================================================================#

start()