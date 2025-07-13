using GLMakie, StaticArrays, LinearAlgebra, DifferentialEquations, Observables
#Initial conditions
const u₀ = SVector{3,Float64}(1.0, 0.0, 0.0)
const Z₀ = SMatrix{3,3,Float64}(I)
const Y₀ = @SVector zeros(9)
const X₀ = SVector{21,Float64}(u₀..., Z₀..., Y₀...)
#adj : X -> adjugate(X)
@inline adj(X) = @SMatrix [X[5]*X[9]-X[6]*X[8] X[3]*X[8]-X[2]*X[9] X[2]*X[6]-X[3]*X[5]; X[6]*X[7]-X[4]*X[9] X[1]*X[9]-X[3]*X[7] X[3]*X[4]-X[1]*X[6]; X[4]*X[8]-X[5]*X[7] X[2]*X[7]-X[1]*X[8] X[1]*X[5]-X[2]*X[4]]
#hat : v -> v̂
@inline hat(v) = @SMatrix [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
#unpackX: X -> (u, Z, Y)
@inline unpackX(X) = SVector{3,Float64}(X[1], X[2], X[3]), SMatrix{3,3}(ntuple(i -> X[3+i], 9)), SMatrix{3,3}(ntuple(i -> X[12+i], 9))
#Computing the geodesic u̇ (Euler-Arnold), the linearized velocity Ż (Linearized Euler-Arnold), and the Jacobi field Ẏ
function F(X, Λ, t) #Benchmark Results: Out-of-Place (SVector) => Trial(6.039 ms), In-Place (MVector) => Trial(7.095 ms)
    u, Z, Y = unpackX(X)
    u̇ = (hat(Λ .* u) * u) ./ Λ                  #u̇ = ad_u⃰(u)
    Ż = (hat(u) * (Λ .* Z)) - (hat(Λ .* u) * Z) #Ż = ad_u⃰(Z) + ad_Z⃰(u)
    Ẏ = Z - (hat(u) * Y)                        #Ẏ = Z - ad_u(Y)
    return SVector{21,Float64}(u̇..., Ż..., Ẏ...)
end 
#The callback will terminate the solver when either of the out variables are equal to 0
function condition(out, X, t, integrator)
    u, Z, Y = unpackX(X)
    Ẏ = Z - (hat(u) * Y)
    out[1] = det(Y)
    out[2] = sum(adj(Y) .* Ẏ)   #d/dt(det(Y)) == 0  here we solve for it by taking the Frobenius inner product of adj(Y) and Ẏ
end                             #this computation involves near singular matrices and forces us to use tighter tolerances / more interpolants to obtain accurate results
affect_pos!(integ, idx) = terminate!(integ)                                                                                     #High interp points allows us to lower the tolerances of the solver this is
affect_neg!(integ, idx) = idx == 1 ? terminate!(integ) : nothing                                                                #desirable because interp points are computationally cheaper than precision
const fzfm = VectorContinuousCallback(condition, affect_pos!, affect_neg!, 2, interp_points=20, save_positions=(false,false))   #The combination of using the DP5() solver, interp_points=20, tol=1e-5, and
const settings = (callback=fzfm, reltol=1e-5, abstol=1e-5, dense=false, save_everystep=false, save_start=false, save_end=true)  #a VectorContinuousCallback is a sweet spot for speed and accuracy
#A CallbackSet is faster, but for some reason it's less accurate than the VectorContinuousCallback with these settings

#========================================================================================================#

output_func(sol, i) = (sol.t[end], sol.retcode == :Terminated)

function createsurface(Λ, res, T)
    nθ, nϕ, sphere = resolutiondict[res]..., spheres[res]
    prob₀ = ODEProblem(F, X₀, (0.0, T), [1.2, 1.0, 0.8])
    prob_func(p, i, repeat) = remake(p; u0=SVector(sphere[i]..., Z₀..., Y₀...))
    ensembleprob = EnsembleProblem(remake(prob₀; p=Λ), prob_func=prob_func, output_func=output_func, safetycopy=false)
    sim = solve(ensembleprob, DP5(), EnsembleThreads(); trajectories=nθ*nϕ, settings...)
    Tmat = reshape(first.(sim.u), nθ, nϕ)
    coords = Tmat .* sphere
    x, y, z = [getindex.(coords, i) for i in 1:3]
    return x, y, z, Tmat
end

createsphere(nθ, nϕ; θ=range(0, π, nθ), ϕ=range(0, 2π, nϕ)') = @. SVector(sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))

const resolutionlist = [:Low, :Medium, :High, :Extreme]
const resolutiondict = Dict(resolutionlist .=> ((120,240), (240,480), (360,720), (960,1920)))
const spheres = Dict(res => createsphere(dims...) for (res, dims) in resolutiondict)

#========================================================================================================#

const requestchannel = Channel{Any}(1)
const surfaceholder = Ref{Any}()
const updatetrigger = Observable{Int}(0)
const surfacecache = Dict{}()

function updatesurface!(Λ, resolution, T)
    if haskey(surfacecache, (Λ, resolution, T)) 
        println("WORKER: Loading cached surface for: (Λ=$Λ, resolution=$resolution, tmax=$T)")
        surface = surfacecache[(Λ, resolution, T)]
    else
        println("WORKER: Creating surface for: (Λ=$Λ, resolution=$resolution, tmax=$T)")
        isready(requestchannel) && return println("WORKER: Cancelled for new request.")
        x, y, z, Tmat = createsurface(Λ, resolution, T)
        surface = x, y, z, Tmat, extrema(Tmat)
        surfacecache[(Λ, resolution, T)] = surface
    end
    surfaceholder.x = surface
    notify(updatetrigger)
    println("WORKER: Done.")
    println(floor((Base.summarysize(surfacecache) / 1024^2)*10)/10," MB")
end

function submitrequest(Λ, resolution, T)
    while isready(requestchannel); take!(requestchannel); println("UI: An old request was cancelled."); end
    println("UI: Submitting request for: (Λ=$Λ, resolution=$resolution, tmax=$T)")
    put!(requestchannel, (Λ, resolution, T))
end

const slidersettings(i) = (label=i, snap=false, format="", height=32, range=-2.5:0.01:2.5, startvalue=1.0, color_inactive=:lightgrey)

function initelements()
    fig = Figure(size=(1400, 900))
    Box(fig[1:4,4], color=RGBf(0.95, 0.95, 0.95), strokecolor=:lightgrey, cornerradius=20, alignmode=Outside(-10), height=625)  #controls area
    cg = GridLayout(fig[1:4,4], default_colgap=0, alignmode=Outside(20));
    lg = GridLayout(cg[1,1:6], alignmode=Outside(20), default_colgap=0)
    sliders = SliderGrid(lg[1:3,1], [slidersettings(i) for i in ["Λ₁","Λ₂","Λ₃"]]...)   #Sliders to control Λ values
    textboxes = [Textbox(lg[i,2], placeholder=" ", validator=Float64, width=45, textcolor=:black, textcolor_placeholder=:black) for i in 1:3]
    rg = GridLayout(cg[2,1:6], alignmode=Outside(20))
    Label(rg[1,2:3], "Resolution:        ", justification=:right, halign=:right, fontsize=16)
    menu = Menu(rg[1,3:4], options=resolutionlist, selection_cell_color_inactive=:lightgrey, width=80, halign=:right)    #Basic resolutionlist menu
    Label(rg[2,2:3], "Custom resolution: ", justification=:right, halign=:right, fontsize=16)   #Custom resolution box
    tb = Textbox(rg[2,4], width=50, placeholder=" ", stored_string="120", validator=Int32, textcolor=:black, textcolor_placeholder=:black)
    gb = Button(rg[4,1:4], label="Generate", buttoncolor=:lightgrey, padding=(12,12,12,12)) #button to force submit request with new settings
    Label(rg[3,2:3], "Max simulation time:", justification=:right, halign=:right, fontsize=16)  #custom max simulation time box
    Tbox = Textbox(rg[3,4], validator=Int32, width=50, placeholder=" ", stored_string="15", textcolor=:black, textcolor_placeholder=:black)
    fg = GridLayout(cg[3,1:6], alignmode=Outside(10))
    sb = Button(fg[1,1:2], label="Save plot as png", buttoncolor=:lightgrey, padding=(20,20,20,20)) #button to save plot as png
    warning = Label(fg[2,1:2], halign=:center, padding=(0,0,0,10), fontsize=15) #warning label to check if max t is enough; lets user know if plot was saved
    colsize!(fig.layout, 1, Auto(true))
    colsize!(fig.layout, 4, Fixed(300))
    return fig, sliders.sliders, textboxes, (tb, gb), warning, Tbox, menu, sb
end

initobservables(sliders) = lift(SVector{3}, [s.value for s in sliders]...), Observable(:Low), Observable(15.0), [Observable(zeros(resolutiondict[:Low]...)) for _ in 1:4]..., Observable((0.0, 20.0))

resfrombox(θ; dims=tryparse(Int32,θ.displayed_string[]).*(1,2), res=Symbol("$(dims[1])x$(dims[2])")) = (resolutiondict[res] = dims; spheres[res] = createsphere(dims...); res)

function initgui()
    fig, sliders, textboxes, customres, warning, Tbox, menu, sb = initelements()
    Λ, resolution, T, x, y, z, Tmat, colorrange = initobservables(sliders)
    submitrequest(Λ[], resolution[], T[])
    updatetextboxes(Λ[], textboxes)
    axis = Axis3(fig[1:4,1:3], aspect=(1,1,1), xlabel="X", ylabel="Y", zlabel="Z", limits=(-13, 13, -13, 13, -13, 13))
    surface!(axis, x, y, z; color=Tmat, colormap=:viridis, colorrange=colorrange) #Plot first surface
    display(fig)

    on(newresolution -> (resolution[] = newresolution; submitrequest(Λ[], newresolution, T[])), menu.selection) #On menu selection update resolution and submit request
    on(newΛ -> updateplot(newΛ, resolution[], T[], Λ[], textboxes), throttle(0.01, Λ))                          #On slider move submit request; Slider is throttled
    on(_ -> (x[], y[], z[], Tmat[], colorrange[]) = surfaceholder.x, updatetrigger)                             #Update plot data When the worker notifies the trigger
    [on(_ -> updatesliders(sliders, textboxes, Λ), throttle(0.1, tb.displayed_string)) for tb in textboxes]     #On textbox edit move sliders to
    on(_ -> submitrequest(Λ[], resfrombox(customres[1]), T[]), customres[2].clicks)                             #submit request when generate is pressed
    on(string -> updateT(T, string, Λ[], resolution[]), throttle(0.1, Tbox.displayed_string))                   #submit request when max time box is edited
    on(_ -> updatewarning(warning, Tmat[], T[]), throttle(0.1, Tmat))                                           #warn user if max time was insufficient to capture surface
    on(_ -> saveplot(Λ[], axis, fig, warning), sb.clicks)                                                       #save plot as png on click
end

updateplot(newΛ, resolution, T, Λ, textboxes) = (submitrequest(ifelse.(newΛ.==0, 1e-2, newΛ), resolution, T); updatetextboxes(Λ, textboxes))

updatetextboxes(Λ, textboxes) = [tb.displayed_string=string(Λ[i]) for (i, tb) in enumerate(textboxes)]

updatesliders(sld, tbx, Λ) = set_close_to!.(sld, @SVector [something(tryparse(Float64, tbx[i].displayed_string[]), Λ[][i]) for i in 1:3])

updateT(T, string, Λ, resolution) = (T[] = something(tryparse(Int, string), 15); submitrequest(Λ, resolution, T[]))

updatewarning(w, Tmat, T) = ((w.text,w.color) = (maximum(Tmat) == T) ? ("Try raising max time", :red) : ("Succesful surface plot", :chartreuse4))

function saveplot(Λ, axis, fig, w)
    try
        fn = "$(Λ[1])_$(Λ[2])_$(Λ[3])__$(floor(axis.azimuth[]*100)/100)_$(floor(axis.elevation[]*100)/100).png"
        save(fn, fig)
        w.text, w.color = "Saved $fn", :chartreuse4
    catch e
        @error "Error saving plot!" exception=(e, catch_backtrace())
    end
end

function start()
    initgui()
    
    @async for (Λ, resolution, T) in requestchannel   #When a request is a made compute surface in low resolution, then the target resolution
        try
            updatesurface!(Λ, :Low, T)
            resolution != :Low && updatesurface!(Λ, resolution, T)
        catch e
            @error "Error in worker task!" exception=(e, catch_backtrace())
        end
    end
end

#========================================================================================================#

start()