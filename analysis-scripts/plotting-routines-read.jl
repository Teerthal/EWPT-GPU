module plotters

using WriteVTK
using HDF5
using Plots
using Statistics
using StatsBase
using FFTW
using LaTeXStrings
using LsqFit
using Glob
using Distributed

include("data-routines-read.jl")
using .data_managers

function resweeper(k1,y1,no_sweeped_points)
    k_0 = minimum(k1)
    k_end = maximum(k1)
    
    k_bins = range(k_0,k_end,length=no_sweeped_points)
    # k_sweep = [mean(k1[1+(i-1)*sweep:1+(i)*sweep]) for i in range(1,(size(k1,1)-1)÷sweep)]
    # B_k_points_sweeped = [mean(y1[1+(i-1)*sweep:1+(i)*sweep]) for i in range(1,(size(k1,1)-1)÷sweep)]
    k_sweep = zeros((size(k_bins,1)-1))
    B_k_points_sweeped = zeros((size(k_bins,1)-1))
    # sweep_err = zeros((size(k_bins,1)-1))

    iter = 1
    for kidx in range(2,size(k_bins,1),step=1)
        ki = k_sweep[kidx]
        k_l = k_bins[kidx-1]
        k_r = k_bins[kidx]
        k_mean = 0.5*(k_l+k_r)
        bin_idxs = findall(k_l.<=k1.<k_r)
        k_sweep[iter] = k_mean
        B_k_points_sweeped[iter] = mean(y1[bin_idxs])
        # sweep_err[iter] = std(y1[bin_idxs])./sqrt(size(bin_idxs,1))
        # println(k_l," ",k_r," ",bin_idxs," ",k_mean," ",mean(y1[bin_idxs]))
        iter=iter+1
    end
    
    return k_sweep,B_k_points_sweeped#,sweep_err
end

function k_mean(EM,k)
    EM=EM[findall(k.<=k_xlim_upper)]
    k=k[findall(k.<=k_xlim_upper)]
    mean_k =sum(k[findall(@. !isnan(EM))].*EM[findall(@. !isnan(EM))])/sum(EM[findall(@. !isnan(EM))])
    return mean_k
end

function xi_mean(EM,k)
    # EM=EM[findall(k.<=k_xlim_upper)]
    # k=k[findall(k.<=k_xlim_upper)]
    e = EM[findall(@. !isnan(EM))]
    K = k[findall(@. !isnan(EM))]
    
    K = K[findall(K.!=0)]
    e = e[findall(K.!=0)]

    mean_k =2*pi*sum(e./K)/sum(e)
    return mean_k
end

function xi_std(EM,k)
    e = EM[findall(@. !isnan(EM))]
    K = k[findall(@. !isnan(EM))]
    
    K = K[findall(K.!=0)]
    e = e[findall(K.!=0)]
    mean_xi_M =2*pi*sum(e./K)/sum(e)
    std_k =sqrt((sum(e.*(2*pi./K.-1.0/mean_xi_M).^2)/sum(e))./size(K,1))
    return std_k
end

function phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)
    k_phys=(B_fft_re[1:end,1])*2*pi/(dx*Nx)
    Bk_phys=4*pi*(k_phys.^2/(dx*Nx)^3).*B_fft_re[1:end,2]
    k_phys_binned=(B_fft_re_binned[1:end,1])*2*pi/(dx*Nx)
    Bk_phys_binned=4*pi*(k_phys_binned.^2/(dx*Nx)^3).*B_fft_re_binned[1:end,2]
    Bk_phys_binned_err=4*pi*(k_phys_binned.^2/(dx*Nx)^3).*B_fft_re_binned[1:end,4]
    return k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err
end

function phys_conv_H(H_fft_re,H_fft_re_binned,dx,Nx)
    Hk_phys=H_fft_re[1:end,2]
    Hk_phys_binned=H_fft_re_binned[1:end,2]
    Hk_phys_binned_err=H_fft_re_binned[1:end,4]
    return Hk_phys,Hk_phys_binned,Hk_phys_binned_err
end

function phys_conv_hel(hel_fft_re,hel_fft_re_binned,dx,Nx)
    helk_phys=hel_fft_re[1:end,2]
    helk_phys_binned=hel_fft_re_binned[1:end,2]
    helk_phys_binned_err=hel_fft_re_binned[1:end,4]
    return helk_phys,helk_phys_binned,helk_phys_binned_err
end

function phys_conv_nok2(B_fft_re,B_fft_re_binned,dx,Nx)
    k_phys=(B_fft_re[1:end,1])*2*pi/(dx*Nx)
    Bk_phys=4*pi*(1/(dx*Nx)^3).*B_fft_re[1:end,2]
    k_phys_binned=(B_fft_re_binned[1:end,1])*2*pi/(dx*Nx)
    Bk_phys_binned=4*pi*(1/(dx*Nx)^3).*B_fft_re_binned[1:end,2]
    Bk_phys_binned_err=4*pi*(1/(dx*Nx)^3).*B_fft_re_binned[1:end,4]
    return k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err
end

function find_mean_peaks(k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,idx,run_idx)
    γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
    Nx = N
   
    k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        

    t = round(time_stamps[idx]*dt*mH,digits=1)
    # println(idx)#;exit()
   
    k_mean_binned = k_mean(Bk_phys_binned,k_phys_binned)
    k_mean_nonbinned = k_mean(Bk_phys,k_phys)
    if size(Bk_phys_binned[findall(@. !isnan(Bk_phys_binned))],1)>0
        k_max = k_phys_binned[argmax(Bk_phys_binned[findall(@. !isnan(Bk_phys_binned))])]
    else
        k_max= 0.0
    end
    xi_mean_binned = xi_mean(Bk_phys_binned,k_phys_binned)
    xi_mean_nonbinned = xi_mean(Bk_phys,k_phys)
   

    ##For the messed up runs where i computed sum instead of mean

    # k_mean_binned = k_mean(Bk_phys_binned./B_fft_re_binned[1:end,3],k_phys_binned)
    # k_mean_nonbinned = k_mean(Bk_phys,k_phys)
    # k_max = k_phys_binned[argmax(Bk_phys_binned./B_fft_re_binned[1:end,3][findall(@. !isnan(Bk_phys_binned./B_fft_re_binned[1:end,3]))])]
    
    # xi_mean_binned = xi_mean(Bk_phys_binned./B_fft_re_binned[1:end,3],k_phys_binned)
    # xi_mean_nonbinned = xi_mean(Bk_phys,k_phys)

    return t,k_mean_nonbinned,k_mean_binned,k_max,xi_mean_nonbinned,xi_mean_binned
end

function find_mean_peaks_less(B_fft_re,B_fft_re_binned,idx,run_idx)
    γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
    Nx = N
   
    k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        

    t = round(time_stamps[idx]*dt*mH,digits=1)
    # println(idx)#;exit()
   
    k_mean_binned = k_mean(Bk_phys_binned,k_phys_binned)
    k_mean_nonbinned = k_mean(Bk_phys,k_phys)
    if size(Bk_phys_binned[findall(@. !isnan(Bk_phys_binned))],1)>0
        k_max = k_phys_binned[argmax(Bk_phys_binned[findall(@. !isnan(Bk_phys_binned))])]
    else
        k_max= 0.0
    end
    xi_mean_binned = xi_mean(Bk_phys_binned,k_phys_binned)
    xi_mean_nonbinned = xi_mean(Bk_phys,k_phys)
   
    return t,k_mean_nonbinned,k_mean_binned,k_max,xi_mean_nonbinned,xi_mean_binned
end


function model(x,p)
    μ_rad=p[1]
    m_rad=p[2]
    T_rad=p[3]
    ω_rad=sqrt.(x.^2.0.+m_rad^2)

    μ_seed=p[4]
    m_seed=p[5]
    T_seed=p[6]
    k0_seed=p[7]
    ω_seed=sqrt.((x.-k0_seed).^2.0.+m_seed^2)

    return (2.0.*ω_rad./(exp.((ω_rad.-μ_rad)./T_rad.-1)))#+(2.0.*x./(exp.((ω_seed.-μ_seed)./T_seed.-1)))
end

export fft_plots_linear
function fft_plots_linear(para_arr;run_idx=1)
    γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
    # idxs=[(no_fft_snaps+1)÷2,no_fft_snaps-max_spec_t_idx]
    idxs=[(10+1),no_fft_snaps-max_spec_t_idx]

    Nx = N

    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:tab10)
    p1=plot([],[],label="")
    p2=plot([],[],label="")
    plot!(p1,[],[],label="non-binned",seriestype=:scatter,c=:black)
    plot!(p1,[],[],label="binned",c=:black)
    plot!(p1,[],[],label=L"$p_{mean}$",c=:black,ls=:dash)
    plot!(p1,[],[],label=L"$m_Ht$",c=:white)

    plot!(p2,[],[],label="non-binned",seriestype=:scatter,c=:black)
    plot!(p2,[],[],label="binned",c=:black)
    plot!(p2,[],[],label=L"$p_{mean}$",c=:black,ls=:dash)
    plot!(p2,[],[],label=L"$m_Ht$",c=:white)

    iter=1
    # for idx in range(2,size(restacked_fft,1),step=1)
    for idx in idxs

        k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(run_dirs[run_idx],idx,run_idx)
        k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        
        k_phys,Hk_phys,k_phys_binned,Hk_phys_binned,Hk_phys_binned_err=phys_conv(H_fft_re,H_fft_re_binned,dx,Nx)

        # ##E_M spectrum plot##
        plot!(p1,k_phys,Bk_phys,label="",c=colors[iter],la=0.5,seriestype=:scatter,ms=1.5,mc=colors[iter],msc=colors[iter],ma=0.5)
        
        plot!(p1,k_phys_binned,Bk_phys_binned,yerr=Bk_phys_binned_err,msc=colors[iter],c=colors[iter],
        label=string(round(time_stamps[idx]*dt*mH,digits=0)),msw=0.25,lw=1,ms=1.5)

        plot!(p1,k_phys_binned,Bk_phys_binned,msc=colors[iter],c=colors[iter],
        label="",lw=1.5)

        vline!(p1,[k_mean(Bk_phys_binned,k_phys_binned)],ls=:dash,c=colors[iter],label="")



        ##E_M spectrum plot##For sum computed specstacks##
        # plot!(p1,k_phys,Bk_phys,label="",c=colors[iter],la=0.5,seriestype=:scatter,ms=1.5,mc=colors[iter],msc=colors[iter],ma=0.5)
        
        # plot!(p1,k_phys_binned,Bk_phys_binned./B_fft_re_binned[1:end,3],yerr=Bk_phys_binned_err./B_fft_re_binned[1:end,3],msc=colors[iter],c=colors[iter],
        # label=string(round(time_stamps[idx]*dt*mH,digits=0)),msw=0.25,lw=1,ms=1.5)

        # plot!(p1,k_phys_binned,Bk_phys_binned./B_fft_re_binned[1:end,3],msc=colors[iter],c=colors[iter],
        # label="",lw=1.5)

        # vline!(p1,[k_mean(Bk_phys_binned./B_fft_re_binned[1:end,3],k_phys_binned)],ls=:dash,c=colors[iter],label="")


        ##H_M spectrum plot##
        plot!(p2,k_phys,Hk_phys,label="",c=colors[iter],la=0.5,seriestype=:scatter,ms=1.5,mc=colors[iter],msc=colors[iter],ma=0.5)
        
        plot!(p2,k_phys_binned,Hk_phys_binned,yerr=Hk_phys_binned_err,msc=colors[iter],c=colors[iter],
        label=string(round(time_stamps[idx]*dt*mH,digits=0)),msw=0.25,lw=1,ms=1.5)

        plot!(p2,k_phys_binned,Hk_phys_binned,msc=colors[iter],c=colors[iter],
        label="",lw=1.5)

        vline!(p2,[k_mean(Hk_phys_binned,k_phys_binned)],ls=:dash,c=colors[iter],label="")

        iter=iter+1

    end
    plot!(p1,xlims=[0,k_xlim_upper],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    plot!(p1,dpi=600)
    plot!(p1,xlabel=L"p/\eta",ylabel=L"4\pi p^2|B(p)|^2/V")
    plot!(p1,minorgrid=true,yscale=:linear,framestyle=:box)
    png(p1,string(run_dirs[run_idx],"/post-spec-t-combined-linear.png"))


    plot!(p2,xlims=[0,k_xlim_upper],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    plot!(p2,dpi=600)
    plot!(p2,xlabel=L"p/\eta",ylabel=L"4\pi p^2|H(p)|^2/V")
    plot!(p2,minorgrid=true,yscale=:linear,framestyle=:box)
    png(p2,string(run_dirs[run_idx],"/post-spec-t-combined-linear-hel.png"))
    

end


export fft_plots_linear_therm
function fft_plots_linear_therm(para_arr;no_runs=25)
    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:tab10)
    p1=plot([],[],label="")
    p2=plot([],[],label="")
    # plot!(p1,[],[],label="non-binned",seriestype=:scatter,c=:black)
    # plot!(p1,[],[],label="binned",c=:black)
    # plot!(p1,[],[],label=L"$p_{mean}$",c=:black,ls=:dash)
    plot!(p1,[],[],label=L"$m_Ht$",c=:white)
    plot!(p2,[],[],label=L"$m_Ht$",c=:white)

    time_plot_idxs = []
    kplot = []
    spec_stack = []
    spec_stack_hel = []

    time_stamps = []
    dt = 0.0

    #320#
    sub_runs_plots = filter(x -> x ∉ [8,22,23], range(1,no_runs,step=1))
    #320#dx:0.1#gamma:0.0
    sub_runs_plots = filter(x -> x ∉ [26], range(1,no_runs,step=1))
    #256 run-dir#gamma:0.01
    sub_runs_plots = filter(x -> x ∉ [5], range(1,no_runs,step=1))
    #256 run-dir#gamma:0.001;T=0.46
    sub_runs_plots = filter(x -> x ∉ [265], range(1,no_runs,step=1))
    #320 run-dir#gamma:0.001;T=0.25
    sub_runs_plots = filter(x -> x ∉ [265], range(1,no_runs,step=1))


    max_EM = 0.0

    for run_idx in sub_runs_plots
        γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
        Nx = N
        idxs=[(10+1),(20+1),no_fft_snaps-max_spec_t_idx]
        # idxs=[(10+1),(20+1),(30+1)]
        # idxs=[(10+1),(14+1)]

        time_stamps = time_stamps
        dt = dt 

        time_plot_idxs = idxs
        # idxs=[(no_fft_snaps+1)÷2,no_fft_snaps-max_spec_t_idx]
        iter=1
        # for idx in range(2,size(restacked_fft,1),step=1)
        spec_stack_t = []
        spec_stack_t_hel = []

        for idx in idxs
            # println(idx)
            # k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(run_dirs[run_idx],idx,run_idx)
            B_fft_re,B_fft_re_binned,H_fft_re,H_fft_re_binned,hel_fft_re,hel_fft_re_binned = load_less_processed_fft_data(run_dirs[run_idx],idx,run_idx)

            k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        
            k_phys,Hk_phys,k_phys_binned,Hk_phys_binned,Hk_phys_binned_err=phys_conv(H_fft_re,H_fft_re_binned,dx,Nx)
            helk_phys,helk_phys_binned,helk_phys_binned_err = phys_conv_hel(hel_fft_re,hel_fft_re_binned,dx,Nx)

            ##E_M spectrum plot##
            plot!(p1,k_phys_binned,Bk_phys_binned,msc=colors[iter],c=colors[iter],
            label="",lw=0.3,alpha=0.5,ls=:dot)
            push!(spec_stack_t,Bk_phys_binned)
            kplot = k_phys_binned

            ##H_M spectrum
            plot!(p2,k_phys_binned,helk_phys_binned,msc=colors[iter],c=colors[iter],
            label="",lw=0.3,alpha=0.5,ls=:dot)
            push!(spec_stack_t_hel,helk_phys_binned)

            iter=iter+1
        end
        push!(spec_stack,spec_stack_t)
        push!(spec_stack_hel,spec_stack_t_hel)
        
    end

    ##ensemble averaged spectra##
    iter = 1
    for idx in time_plot_idxs

        # println(size([spec_stack[i][iter] for i in range(1,no_runs,step=1)]))
        # plot!(p1,kplot,mean([spec_stack[i][iter] for i in range(1,no_runs,step=1)]),yerr=std([spec_stack[i][iter] for i in range(1,no_runs,step=1)])./sqrt(no_runs),msc=colors[iter],c=colors[iter],
        # label="",lw=1.5)

        ###E_M spectra###
        mean_EM_t = mean([spec_stack[i][iter] for i in range(1,size(spec_stack,1),step=1)])
        std_EM_t = std([spec_stack[i][iter] for i in range(1,size(spec_stack,1),step=1)])/sqrt.(no_runs)

        plot!(p1,kplot,mean_EM_t,yerr=std_EM_t, msc=colors[iter],c=colors[iter],
        label=string(round(time_stamps[idx]*dt*mH,digits=0)),lw=1.5)
        ##################

        ###H_M spectra###
        mean_HM_t = mean([spec_stack_hel[i][iter] for i in range(1,size(spec_stack_hel,1),step=1)])
        std_HM_t = std([spec_stack_hel[i][iter] for i in range(1,size(spec_stack_hel,1),step=1)])/sqrt.(no_runs)

        plot!(p2,kplot,mean_HM_t,yerr=std_HM_t, msc=colors[iter],c=colors[iter],
        label=string(round(time_stamps[idx]*dt*mH,digits=0)),lw=1.5)

        ##################

        iter = iter+1

        if maximum(mean_EM_t)> max_EM
            max_EM=maximum(mean_EM_t)
        end

    end
    # plot!(p1,xlims=[0,k_xlim_upper],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    plot!(p1,xlims=[0,3],ylims = [0,max_EM*1.2],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)

    plot!(p1,dpi=600)
    plot!(p1,xlabel=L"p/\eta",ylabel=L"4\pi p^2|B(p)|^2/V")
    plot!(p1,minorgrid=true,yscale=:linear,framestyle=:box)
    png(p1,string(string(master_dir,"/",dir_name(para_arr[1])),"/post-spec-t-combined-linear-ensemble.png"))

    plot!(p2,dpi=600)
    plot!(p2,xlabel=L"p/\eta",ylabel=L"H_M")
    plot!(p2,minorgrid=true,yscale=:linear,framestyle=:box)
    # png(p2,string(string(master_dir,"/",dir_name(para_arr[1])),"/post-hel-spec-t-combined-linear-ensemble.png"))
    savefig(p2,string(string(master_dir,"/",dir_name(para_arr[1])),"/post-hel-spec-t-combined-linear-ensemble.pdf"))

end

export helicity_fft_plots_therm
function helicity_fft_plots_therm(para_arr;no_runs=25)
    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:tab10)
    p1=plot([],[],label="")
    p2=plot([],[],label="")
    # plot!(p1,[],[],label="non-binned",seriestype=:scatter,c=:black)
    # plot!(p1,[],[],label="binned",c=:black)
    # plot!(p1,[],[],label=L"$p_{mean}$",c=:black,ls=:dash)
    plot!(p1,[],[],label=L"$m_Ht$",c=:white)
    plot!(p2,[],[],label=L"$m_Ht$",c=:white)

    time_plot_idxs = []
    kplot = []
    spec_stack = []
    spec_stack_hel = []

    time_stamps = []
    dt = 0.0
    min_k=100.0
    
    #320#
    sub_runs_plots = filter(x -> x ∉ [8,22,23], range(1,no_runs,step=1))
    #320#dx:0.1#gamma:0.0
    sub_runs_plots = filter(x -> x ∉ [26], range(1,no_runs,step=1))
    #256 run-dir#gamma:0.01
    # sub_runs_plots = filter(x -> x ∉ [5], range(1,no_runs,step=1))
    ##256 run-dir#dx=0.1#dt=dx/20#nte:320000#
    sub_runs_plots = filter(x -> x ∉ [5], range(1,no_runs,step=1))
    #320 run-dir#gamma:0.001;T=0.25
    sub_runs_plots = filter(x -> x ∉ [265], range(1,no_runs,step=1))

    max_EM = 0.0

    for run_idx in sub_runs_plots
        γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
        Nx = N
        idxs=[(10+1),(20+1),no_fft_snaps-max_spec_t_idx]
        # idxs=[(10+1),(20+1),(30+1)]

        time_stamps = time_stamps
        dt = dt 

        time_plot_idxs = idxs
        # idxs=[(no_fft_snaps+1)÷2,no_fft_snaps-max_spec_t_idx]
        iter=1
        # for idx in range(2,size(restacked_fft,1),step=1)
        spec_stack_t = []
        spec_stack_t_hel = []

        for idx in idxs

            # k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(run_dirs[run_idx],idx,run_idx)
            B_fft_re,B_fft_re_binned,H_fft_re,H_fft_re_binned,hel_fft_re,hel_fft_re_binned = load_less_processed_fft_data(run_dirs[run_idx],idx,run_idx)

            k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        
            k_phys,Hk_phys,k_phys_binned,Hk_phys_binned,Hk_phys_binned_err=phys_conv(H_fft_re,H_fft_re_binned,dx,Nx)
            helk_phys,helk_phys_binned,helk_phys_binned_err = phys_conv_hel(hel_fft_re,hel_fft_re_binned,dx,Nx)

            # kplot = k_phys_binned
            kplot = k_phys_binned[findall(k_phys_binned.!=0)]

            ##H_M spectrum
            plot!(p2,kplot,abs.(helk_phys_binned[findall(k_phys_binned.!=0)]),msc=colors[iter],c=colors[iter],
            label="",lw=0.3,alpha=0.5,ls=:dot)
            push!(spec_stack_t_hel,helk_phys_binned[findall(k_phys_binned.!=0)])

            iter=iter+1

            if minimum(kplot)<=min_k
                min_k=minimum(kplot)
            end

            if (any(isnan.(helk_phys_binned[findall(k_phys_binned.!=0)])))
                println(string("broken run:",run_idx," at ",idx))
                exit()
            end

        end
        push!(spec_stack_hel,spec_stack_t_hel)
        
    end

    ##ensemble averaged spectra##
    iter = 1
    for idx in time_plot_idxs

        ###H_M spectra###
        mean_HM_t = mean([spec_stack_hel[i][iter] for i in range(1,size(spec_stack_hel,1),step=1)])
        std_HM_t = std([spec_stack_hel[i][iter] for i in range(1,size(spec_stack_hel,1),step=1)])/sqrt.(no_runs)

        plot!(p2,kplot,abs.(mean_HM_t),yerr=std_HM_t, msc=colors[iter],c=colors[iter],
        label=string(round(time_stamps[idx]*dt*mH,digits=0)),lw=1.5)

        ##################

        iter = iter+1

        # if maximum(mean_EM_t)> max_EM
        #     max_EM=maximum(mean_EM_t)
        # end

    end
    plot!(p2,dpi=600)
    plot!(p2,xlabel=L"p/\eta",ylabel=L"H_M")
    plot!(p2,minorgrid=true,yscale=:linear,framestyle=:box)
    # plot!(p2,xscale=:log,yscale=:log)
    # plot!(p2,yscale=:log)
    plot!(p2,xlims=[min_k/1.1,k_xlim_upper/1.5],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    # png(p2,string(string(master_dir,"/",dir_name(para_arr[1])),"/post-hel-spec-t-combined-linear-ensemble.png"))
    savefig(p2,string(string(master_dir,"/",dir_name(para_arr[1])),"/post-hel-spec-details-t-combined-linear-ensemble.pdf"))

end

function hosking(;idxs=[(no_fft_snaps+1)÷2,no_fft_snaps])

    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:tab10)
    p1=plot([],[],label="")
    p2=plot([],[],label="")
    plot!(p1,[],[],label="non-binned",seriestype=:scatter,c=:black)
    plot!(p1,[],[],label="binned",c=:black)
    plot!(p1,[],[],label=L"$p_{mean}$",c=:black,ls=:dash)
    plot!(p1,[],[],label=L"$m_Ht$",c=:white)

    plot!(p2,[],[],label="non-binned",seriestype=:scatter,c=:black)
    plot!(p2,[],[],label="binned",c=:black)
    plot!(p2,[],[],label=L"$p_{mean}$",c=:black,ls=:dash)
    plot!(p2,[],[],label=L"$m_Ht$",c=:white)

    iter=1
    # for idx in range(2,size(restacked_fft,1),step=1)
    for idx in idxs

        k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(idx)
        k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        
        k_phys,Hk_phys,k_phys_binned,Hk_phys_binned,Hk_phys_binned_err=phys_conv(H_fft_re,H_fft_re_binned,dx,Nx)

        I_H = 

        ##E_M spectrum plot##
        plot!(p1,k_phys,Bk_phys,label="",c=colors[iter],la=0.5,seriestype=:scatter,ms=1.5,mc=colors[iter],msc=colors[iter],ma=0.5)
        
        plot!(p1,k_phys_binned,Bk_phys_binned,yerr=Bk_phys_binned_err,msc=colors[iter],c=colors[iter],
        label=string(round(time_stamps[idx]*dt*mH,digits=0)),msw=0.25,lw=1,ms=1.5)

        plot!(p1,k_phys_binned,Bk_phys_binned,msc=colors[iter],c=colors[iter],
        label="",lw=1.5)

        vline!(p1,[k_mean(Bk_phys_binned,k_phys_binned)],ls=:dash,c=colors[iter],label="")

        ##H_M spectrum plot##
        plot!(p2,k_phys,Hk_phys,label="",c=colors[iter],la=0.5,seriestype=:scatter,ms=1.5,mc=colors[iter],msc=colors[iter],ma=0.5)
        
        plot!(p2,k_phys_binned,Hk_phys_binned,yerr=Hk_phys_binned_err,msc=colors[iter],c=colors[iter],
        label=string(round(time_stamps[idx]*dt*mH,digits=0)),msw=0.25,lw=1,ms=1.5)

        plot!(p2,k_phys_binned,Hk_phys_binned,msc=colors[iter],c=colors[iter],
        label="",lw=1.5)

        vline!(p2,[k_mean(Hk_phys_binned,k_phys_binned)],ls=:dash,c=colors[iter],label="")

        iter=iter+1

    end
    plot!(p1,xlims=[0,k_xlim_upper],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    plot!(p1,dpi=600)
    plot!(p1,xlabel=L"p/\eta",ylabel=L"4\pi p^2|B(p)|^2/V")
    plot!(p1,minorgrid=true,yscale=:linear,framestyle=:box)
    png(p1,string(run_dir,"/post-spec-t-combined-linear.png"))


    plot!(p2,xlims=[0,k_xlim_upper],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    plot!(p2,dpi=600)
    plot!(p2,xlabel=L"p/\eta",ylabel=L"4\pi p^2|H(p)|^2/V")
    plot!(p2,minorgrid=true,yscale=:linear,framestyle=:box)
    png(p2,string(run_dir,"/post-spec-t-combined-linear-hel.png"))
    

end

function fft_plots_linear_nok2(;idxs=[(no_fft_snaps+1)÷2,no_fft_snaps])

    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:tab10)
    p1=plot([],[],label="")
    p2=plot([],[],label="")
    plot!(p1,[],[],label="non-binned",seriestype=:scatter,c=:black)
    plot!(p1,[],[],label="binned",c=:black)
    plot!(p1,[],[],label=L"$p_{mean}$",c=:black,ls=:dash)
    plot!(p1,[],[],label=L"$m_Ht$",c=:white)

    iter=1
    
    for idx in idxs
        # k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned=load_processed_fft_data(idx)
        k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(idx)
        k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv_nok2(B_fft_re,B_fft_re_binned,dx,Nx)        

        plot!(p1,k_phys,Bk_phys,label="",c=colors[iter],la=0.5,seriestype=:scatter,ms=1.5,mc=colors[iter],msc=colors[iter],ma=0.5)
        
        plot!(p1,k_phys_binned,Bk_phys_binned,yerr=Bk_phys_binned_err,msc=colors[iter],c=colors[iter],
        label=string(round(time_stamps[idx]*dt*mH,digits=0)),msw=0.25,lw=1,ms=1.5)

        plot!(p1,k_phys_binned,Bk_phys_binned,msc=colors[iter],c=colors[iter],
        label="",lw=1.5)

        # vline!(p1,[k_mean(Bk_phys,k_phys)],ls=:dot,c=colors[iter])
        vline!(p1,[k_mean(Bk_phys_binned,k_phys_binned)],ls=:dash,c=colors[iter],label="")

        # if iter==1
        #     plot!(p2,k_phys_binned,B_fft_re_binned[1:end,3],seriestype=:bar,linecolor=nothing,label="",bottom_margin=0*Plots.mm)
        # end


        iter=iter+1

    end
    plot!(p1,xlims=[0,k_xlim_upper],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    plot!(p1,dpi=600)
    plot!(p1,xlabel=L"p/\eta",ylabel=L"4\pi |B(p)|^2/V")
    # plot!(p2,ylabel="#modes")

    plot!(p1,minorgrid=true,yscale=:linear,framestyle=:box)
    # plot!(p2,xlims=[0,k_xlim_upper],ylims=[0,10000],framestyle=:box)
    # plot(p2,p1,layout=grid(2, 1, heights=[0.2, 0.8]),dpi=600)
    png(p1,string(run_dir,"/post-spec-t-combined-linear-nok2.png"))

end

function fft_plots_log_nok2(;idxs=[(no_fft_snaps+1)÷2,no_fft_snaps])

    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:tab10)
    p1=plot([],[],label="")
    p2=plot([],[],label="")
    plot!(p1,[],[],label="non-binned",seriestype=:scatter,c=:black)
    plot!(p1,[],[],label="binned",c=:black)
    plot!(p1,[],[],label=L"$p_{mean}$",c=:black,ls=:dash)
    plot!(p1,[],[],label=L"$m_Ht$",c=:white)

    plot!(p2,[],[],label="non-binned",seriestype=:scatter,c=:black)
    plot!(p2,[],[],label="binned",c=:black)
    plot!(p2,[],[],label=L"$p_{mean}$",c=:black,ls=:dash)
    plot!(p2,[],[],label=L"$m_Ht$",c=:white)

    min_k=100.0
    iter=1
    # for idx in range(2,size(restacked_fft,1),step=1)
    for idx in idxs

        # k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned=load_processed_fft_data(idx)
        k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(idx)
        k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv_nok2(B_fft_re,B_fft_re_binned,dx,Nx)        
        k_phys,Hk_phys,k_phys_binned,Hk_phys_binned,Hk_phys_binned_err=phys_conv(H_fft_re,H_fft_re_binned,dx,Nx)

        plot!(p1,k_phys[findall(k_phys.!=0)],Bk_phys[findall(k_phys.!=0)],label="",c=colors[iter],la=0.5,seriestype=:scatter,ms=1.5,mc=colors[iter],msc=colors[iter],ma=0.5)
        
        plot!(p1,k_phys_binned,Bk_phys_binned,yerr=Bk_phys_binned_err,msc=colors[iter],c=colors[iter],
        label=string(round(time_stamps[idx]*dt*mH,digits=0)),msw=0.25,lw=1,ms=1.5)

        plot!(p1,k_phys_binned,Bk_phys_binned,msc=colors[iter],c=colors[iter],
        label="",lw=1.5)

        vline!(p1,[k_mean(Bk_phys_binned,k_phys_binned)],ls=:dash,c=colors[iter],label="")

        # if iter==1
        #     plot!(p2,k_phys_binned,B_fft_re_binned[1:end,3],seriestype=:bar,linecolor=nothing,label="",bottom_margin=0*Plots.mm)
        # end

        plot!(p2,k_phys[findall(k_phys.!=0)],Hk_phys[findall(k_phys.!=0)],label="",c=colors[iter],la=0.5,seriestype=:scatter,ms=1.5,mc=colors[iter],msc=colors[iter],ma=0.5)
        
        # plot!(p2,k_phys_binned,Hk_phys_binned,yerr=Hk_phys_binned_err,msc=colors[iter],c=colors[iter],
        # label=string(round(time_stamps[idx]*dt*mH,digits=0)),msw=0.25,lw=1,ms=1.5)

        # plot!(p2,k_phys_binned,Hk_phys_binned,msc=colors[iter],c=colors[iter],
        # label="",lw=1.5)

        vline!(p2,[k_mean(Hk_phys_binned,k_phys_binned)],ls=:dash,c=colors[iter],label="")


        if minimum(k_phys[findall(k_phys.!=0)])<=min_k
            min_k=minimum(k_phys[findall(k_phys.!=0)])
        end

        iter=iter+1

    end
    plot!(p1,xlims=[min_k,k_xlim_upper],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    plot!(p1,xscale=:log,yscale=:log)
    plot!(p1,dpi=600)
    plot!(p1,xlabel=L"p/\eta",ylabel=L"4\pi |B(p)|^2/V")
    
    png(p1,string(run_dir,"/post-spec-t-combined-log-nok2.png"))

    plot!(p2,xlims=[min_k,k_xlim_upper],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    plot!(p2,xscale=:log,yscale=:log)
    plot!(p2,dpi=600)
    plot!(p2,xlabel=L"p/\eta",ylabel=L"4\pi |H(p)|^2/V")
    
    png(p2,string(run_dir,"/post-spec-t-combined-log-nok2-hel.png"))


end

export fft_plots_log
function fft_plots_log(para_arr;run_idx=1)
    γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])

    # idxs=[(no_fft_snaps+1)÷2,no_fft_snaps-max_spec_t_idx]
    idxs=[(10+1),no_fft_snaps-max_spec_t_idx]

    Nx = N

    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:tab10)
    p1=plot([],[],label="")
    p2=plot([],[],label="")
    plot!(p1,[],[],label="non-binned",seriestype=:scatter,c=:black)
    plot!(p1,[],[],label="binned",c=:black)
    plot!(p1,[],[],label=L"$p_{mean}$",c=:black,ls=:dash)
    plot!(p1,[],[],label=L"$(p/\eta)^\alpha$",c=:black,ls=:dashdot)
    plot!(p1,[],[],label=L"$m_Ht$",c=:white)

    plot!(p2,[],[],label="non-binned",seriestype=:scatter,c=:black)
    plot!(p2,[],[],label="binned",c=:black)
    plot!(p2,[],[],label=L"$p_{mean}$",c=:black,ls=:dash)
    plot!(p2,[],[],label=L"$m_Ht$",c=:white)

    iter=1
    # for idx in range(2,size(restacked_fft,1),step=1)
    min_k = 100.0
    min_B_k = 1.0
    min_H_k = 1.0
    max_B_k = 1.0
    max_H_k = 1.0
    max_pad_factor = 2.0

    for idx in idxs

        # k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned=load_processed_fft_data(idx)
        k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(run_dirs[run_idx],idx,run_idx)
        k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        
        k_phys,Hk_phys,k_phys_binned,Hk_phys_binned,Hk_phys_binned_err=phys_conv(H_fft_re,H_fft_re_binned,dx,Nx)

        plot!(p1,k_phys[findall(k_phys.!=0)],Bk_phys[findall(k_phys.!=0)],label="",c=colors[iter],la=0.5,seriestype=:scatter,ms=1.5,mc=colors[iter],msc=colors[iter],ma=0.5)
        
        plot!(p1,k_phys_binned[findall(k_phys_binned.!=0)],Bk_phys_binned[findall(k_phys_binned.!=0)],yerr=Bk_phys_binned_err[findall(k_phys_binned.!=0)],msc=colors[iter],c=colors[iter],
        label=string(round(time_stamps[idx]*dt*mH,digits=0)),msw=0.25,lw=1,ms=1.5)

        plot!(p1,k_phys_binned[findall(k_phys_binned.!=0)],Bk_phys_binned[findall(k_phys_binned.!=0)],msc=colors[iter],c=colors[iter],
        label="",lw=1.5)

        vline!(p1,[k_mean(Bk_phys_binned,k_phys_binned)],ls=:dash,c=colors[iter],label="")
        vline!(p1,[k_phys_binned[findall(Bk_phys_binned.==maximum(Bk_phys_binned))]],ls=:dot,c=colors[iter],label="")
        # if iter==1
        #     plot!(p2,k_phys_binned,B_fft_re_binned[1:end,3],seriestype=:bar,linecolor=nothing,label="",bottom_margin=0*Plots.mm)
        # end

        m(t, p) = p[1]*t.+p[2]

        ##low_k fit
        p0 = [4.0, 1.0]

        low_k_cut_idx = findall(Bk_phys_binned.==maximum(Bk_phys_binned))[1]
        fit = curve_fit(m, log.(k_phys_binned[1:low_k_cut_idx]), log.(Bk_phys_binned[1:low_k_cut_idx]), p0)
        plot!(p1,k_phys_binned[1:low_k_cut_idx],exp.(m(log.(k_phys_binned[1:low_k_cut_idx]),fit.param)),
        ls=:dashdot,c=colors[iter],label=string(L"$\alpha_1$:",round(fit.param[1],digits=1)))

        ##High k fit
        p0 = [-4.0, 1.0]

        high_k_cut_idx = findall(Bk_phys_binned.==maximum(Bk_phys_binned))[1]
        high_k_cut_upp_idx =(findall(Bk_phys_binned.>maximum(Bk_phys_binned/1e1)))[end]

        fit = curve_fit(m, log.(k_phys_binned[high_k_cut_idx:high_k_cut_upp_idx]), log.(Bk_phys_binned[high_k_cut_idx:high_k_cut_upp_idx]), p0)
        plot!(p1,k_phys_binned[high_k_cut_idx:high_k_cut_upp_idx],exp.(m(log.(k_phys_binned[high_k_cut_idx:high_k_cut_upp_idx]),fit.param)),
        ls=:dashdot,c=colors[iter*3],label=string(L"$\alpha_2$:",round(fit.param[1],digits=1)))
        
        vline!(p1,[k_phys_binned[high_k_cut_upp_idx]],c=colors[iter*3],ls=:dashdot,label="")

        ###Very high k fit###
        veryhigh_k_cut_upp_idx = (findall(Bk_phys_binned.>maximum(Bk_phys_binned/1e4)))[end]
        fit = curve_fit(m, log.(k_phys_binned[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]), log.(Bk_phys_binned[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]), p0)
        plot!(p1,k_phys_binned[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx],exp.(m(log.(k_phys_binned[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]),fit.param)),
        ls=:dashdot,c=colors[iter*4],label=string(L"$\alpha_3$:",round(fit.param[1],digits=1)))
        
        vline!(p1,[k_phys_binned[high_k_cut_upp_idx]],c=colors[iter*4],ls=:dashdot,label="")
        

        ##plotting hel-spectra##
        plot!(p2,k_phys[findall(k_phys.!=0)],Hk_phys[findall(k_phys.!=0)],label="",c=colors[iter],la=0.5,seriestype=:scatter,ms=1.5,mc=colors[iter],msc=colors[iter],ma=0.5)
        
        plot!(p2,k_phys_binned[findall(k_phys_binned.!=0)],Hk_phys_binned[findall(k_phys_binned.!=0)],yerr=Hk_phys_binned_err[findall(k_phys_binned.!=0)],msc=colors[iter],c=colors[iter],
        label=string(round(time_stamps[idx]*dt*mH,digits=0)),msw=0.25,lw=1,ms=1.5)

        plot!(p2,k_phys_binned[findall(k_phys_binned.!=0)],Hk_phys_binned[findall(k_phys_binned.!=0)],msc=colors[iter],c=colors[iter],
        label="",lw=1.5)

        vline!(p2,[k_mean(Hk_phys_binned,k_phys_binned)],ls=:dash,c=colors[iter],label="")

        if minimum(k_phys[findall(k_phys.!=0)])<=min_k
            min_k=minimum(k_phys[findall(k_phys.!=0)])
        end

        if minimum(Bk_phys_binned[findall(k_phys_binned.!=0)])<min_B_k
            min_B_k=minimum(Bk_phys_binned[findall(k_phys_binned.!=0)])
        end

        if minimum(Hk_phys_binned[findall(k_phys_binned.!=0)])<min_H_k
            min_B_k=minimum(Hk_phys_binned[findall(k_phys_binned.!=0)])
        end

        if maximum(Bk_phys_binned[findall(k_phys_binned.!=0)])*max_pad_factor>max_B_k
            max_B_k=maximum(Bk_phys_binned[findall(k_phys_binned.!=0)])*max_pad_factor
        end

        if maximum(Hk_phys_binned[findall(k_phys_binned.!=0)])*max_pad_factor>max_H_k
            max_H_k=maximum(Hk_phys_binned[findall(k_phys_binned.!=0)])*max_pad_factor
        end

        iter=iter+1

    end
    plot!(p1,xlims=[min_k,k_xlim_upper],ylims=[max_B_k/1e2,max_B_k],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    plot!(p1,xscale=:log,yscale=:log)
    plot!(p1,dpi=600)
    plot!(p1,xlabel=L"p/\eta",ylabel=L"4\pi p^2|B(p)|^2/V")
    plot!(p1,minorgrid=true,framestyle=:box)
    png(p1,string(run_dirs[run_idx],"/post-spec-t-combined-log.png"))

    plot!(p2,xlims=[min_k,k_xlim_upper],ylims=[max_H_k/1e6,max_H_k],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    plot!(p2,xscale=:log,yscale=:log)
    plot!(p2,dpi=600)
    plot!(p2,xlabel=L"p/\eta",ylabel=L"4\pi p^2|H(p)|^2/V")
    plot!(p2,minorgrid=true,framestyle=:box)
    png(p2,string(run_dirs[run_idx],"/post-spec-t-combined-log-hel.png"))

end

function good_runs(para_arr;run_idx=1)
    γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
    if N==320
        sub_runs_plots = filter(x -> x ∉ [8,22,23], range(1,no_runs,step=1))
    elseif N==256
        sub_runs_plots = filter(x -> x ∉ [8,22,23], range(1,no_runs,step=1))
    end
end
return


export fft_plots_log_therm_detail
function fft_plots_log_therm_detail(para_arr;no_runs=25)
    
    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:hawaii25)
    colors=palette(:seaborn_bright)

    p1=plot([],[],label="")
    p2=plot([],[],label="")
    # plot!(p1,[],[],label="non-binned",seriestype=:scatter,c=:black)
    # plot!(p1,[],[],label="binned",c=:black)
    # plot!(p1,[],[],label=L"$p_{mean}$",c=:black,ls=:dash)
    # plot!(p1,[],[],label=L"fit ranges",c=:black,ls=:dash)
    plot!(p1,[],[],label=L"$m_Ht$",c=:white)

    time_plot_idxs = []
    kplot = []
    spec_stack = []

    time_stamps = []
    dt = 0.0 
    #320#
    sub_runs_plots = filter(x -> x ∉ [8,22,23], range(1,no_runs,step=1))
    #256#
    # sub_runs_plots = filter(x -> x ∉ [17], range(1,no_runs,step=1))
    # sub_runs_plots = filter(x -> x ∉ [17,10,2,6,8,11,12,22,25], range(1,no_runs,step=1))
    #320#dx:0.1#gamma:0.0
    sub_runs_plots = filter(x -> x ∉ [26], range(1,no_runs,step=1))

    for run_idx in sub_runs_plots
        γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
        Nx = N

        time_stamps = time_stamps
        dt = dt 

        # idxs=[(10+1),no_fft_snaps-max_spec_t_idx]
        idxs=[no_fft_snaps-max_spec_t_idx]
        
        #320#
        # idxs=[(10+1),(20+1),(30+1)]
        
        time_plot_idxs = idxs 
        # idxs=[(no_fft_snaps+1)÷2,no_fft_snaps-max_spec_t_idx]
        iter=1
        # for idx in range(2,size(restacked_fft,1),step=1)
        spec_stack_t = []

        # println(time_plot_idxs)
        # println(time_plot_idxs*dsnaps_fft*dt*mH);exit()

        for idx in time_plot_idxs

            # k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(run_dirs[run_idx],idx,run_idx)

            B_fft_re,B_fft_re_binned,H_fft_re,H_fft_re_binned = load_less_processed_fft_data(run_dirs[run_idx],idx,run_idx)

            k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        
            k_phys,Hk_phys,k_phys_binned,Hk_phys_binned,Hk_phys_binned_err=phys_conv(H_fft_re,H_fft_re_binned,dx,Nx)

            ##E_M spectrum plot##
            plot!(p1,k_phys_binned[findall(k_phys_binned.!=0)],Bk_phys_binned[findall(k_phys_binned.!=0)],msc=colors[iter],c=colors[iter],
            label="",lw=0.25,alpha=0.5,ls=:dot)
            push!(spec_stack_t,Bk_phys_binned[findall(k_phys_binned.!=0)])
            kplot = k_phys_binned[findall(k_phys_binned.!=0)]
            iter=iter+1
            if (any(isnan.(Bk_phys_binned[findall(k_phys_binned.!=0)])))
                println(string("broken run:",run_idx," at ",idx))
                exit()
            end
        end
        push!(spec_stack,spec_stack_t)
        # plot!(p1,xscale=:log,yscale=:log)
        # png(p1,string(string(master_dir,"/",dir_name(para_arr[1])),"/test-log",run_idx,"-ensemble.png"))
    end

    ##ensemble averaged spectra##
    iter = 1

    min_k = 100.0
    min_B_k = 1.0
    min_H_k = 1.0
    max_B_k = 1.0
    max_H_k = 1.0
    max_pad_factor = 1.5

    for idx in time_plot_idxs

        Bk_phys_binned_mean = mean([spec_stack[i][iter] for i in range(1,size(spec_stack,1),step=1)])
        Bk_phys_binned_err = std([spec_stack[i][iter] for i in range(1,size(spec_stack,1),step=1)])/sqrt.(no_runs)

        kplot=kplot[findall(kplot.!=0)]
        Bk_phys_binned_mean = Bk_phys_binned_mean[findall(kplot.!=0)]
        Bk_phys_binned_err = Bk_phys_binned_err[findall(kplot.!=0)]

        plot!(p1,kplot,Bk_phys_binned_mean,yerr=Bk_phys_binned_err,msc=colors[iter],c=colors[iter],
        label=string(round(time_stamps[idx]*dt*mH,digits=0)),lw=1.5)

        if idx==time_plot_idxs[end]
            m(t, p) = p[1]*t.+p[2]

            ##low_k fit
            p0 = [4.0, 1.0]

            low_k_cut_idx = findall(Bk_phys_binned_mean.==maximum(Bk_phys_binned_mean))[1]
            fit = curve_fit(m, log.(kplot[1:low_k_cut_idx]), log.(Bk_phys_binned_mean[1:low_k_cut_idx]), p0)
            plot!(p1,kplot[1:low_k_cut_idx],exp.(m(log.(kplot[1:low_k_cut_idx]),fit.param)),
            ls=:dashdot,c=colors[iter],label=string(L"$\alpha_1$:",round(fit.param[1],digits=1)))
            # vline!(p1,[kplot[low_k_cut_idx]],c=:black,ls=:dash,label="")

            ##High k fit
            p0 = [-4.0, 1.0]

            high_k_cut_idx = findall(Bk_phys_binned_mean.==maximum(Bk_phys_binned_mean))[1]
            high_k_cut_upp_idx =(findall(Bk_phys_binned_mean.>maximum(Bk_phys_binned_mean)/3))[end]
            fit = curve_fit(m, log.(kplot[high_k_cut_idx:high_k_cut_upp_idx]), log.(Bk_phys_binned_mean[high_k_cut_idx:high_k_cut_upp_idx]), p0)
            plot!(p1,kplot[high_k_cut_idx:high_k_cut_upp_idx],exp.(m(log.(kplot[high_k_cut_idx:high_k_cut_upp_idx]),fit.param)),
            ls=:dashdot,c=colors[iter],label=string(L"$\alpha_2$:",round(fit.param[1],digits=1)))
            
            # vline!(p1,[kplot[high_k_cut_upp_idx]],c=:red,ls=:dash,label="")

            ###Very high k fit###
            high_k_cut_upp_idx =(findall(Bk_phys_binned_mean.>maximum(Bk_phys_binned_mean)/10))[end]
            veryhigh_k_cut_upp_idx = (findall(Bk_phys_binned_mean.>maximum(Bk_phys_binned_mean/1e3)))[end]
            fit = curve_fit(m, log.(kplot[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]), log.(Bk_phys_binned_mean[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]), p0)
            plot!(p1,kplot[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx],exp.(m(log.(kplot[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]),fit.param)),
            ls=:dashdot,c=colors[iter],label=string(L"$\alpha_3$:",round(fit.param[1],digits=1)))
            
            # vline!(p1,[kplot[veryhigh_k_cut_upp_idx]],c=:blue,ls=:dash,label="")
        end

            iter = iter+1

            if minimum(kplot)<=min_k
                min_k=minimum(kplot)
            end

            if minimum(Bk_phys_binned_mean)<min_B_k
                min_B_k=minimum(Bk_phys_binned_mean)
            end

            if maximum(Bk_phys_binned_mean)*max_pad_factor>max_B_k
                max_B_k=maximum(Bk_phys_binned_mean)*max_pad_factor
            
            end
    end
    
    plot!(p1,xlims=[min_k/1.2,k_xlim_upper],ylims=[max_B_k/100,max_B_k],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    plot!(p1,xscale=:log,yscale=:log)
    plot!(p1,dpi=600)
    plot!(p1,xlabel=L"p/\eta",ylabel=L"4\pi p^2|B(p)|^2/V")
    plot!(p1,minorgrid=true,framestyle=:box)
    # png(p1,string(string(master_dir,"/",dir_name(para_arr[1])),"/post-spec-t-combined-log-ensemble.png"))

    png(p1,string(string(master_dir,"/",dir_name(para_arr[1])),"/post-spec-t-combined-log-ensemble.png"))

end

export fft_plots_log_therm
function fft_plots_log_therm(para_arr;no_runs=25)
    
    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:hawaii25)
    colors=palette(:seaborn_bright)

    p1=plot([],[],label="")
    p2=plot([],[],label="")
    # plot!(p1,[],[],label="non-binned",seriestype=:scatter,c=:black)
    # plot!(p1,[],[],label="binned",c=:black)
    # plot!(p1,[],[],label=L"$p_{mean}$",c=:black,ls=:dash)
    # plot!(p1,[],[],label=L"fit ranges",c=:black,ls=:dash)
    plot!(p1,[],[],label=L"$m_Ht$",c=:white)

    time_plot_idxs = []
    kplot = []
    spec_stack = []

    time_stamps = []
    dt = 0.0 
    #320#
    sub_runs_plots = filter(x -> x ∉ [8,22,23], range(1,no_runs,step=1))
    #256#
    # sub_runs_plots = filter(x -> x ∉ [17,10,2,6,8,11,12,22,25], range(1,no_runs,step=1))
    #320#dx:0.1#gamma:0.0
    sub_runs_plots = filter(x -> x ∉ [1,2,3,4,14,17,21,24], range(1,no_runs,step=1))
    #256 run-dir#gamma:0.01
    sub_runs_plots = filter(x -> x ∉ [5], range(1,no_runs,step=1))
    #256 run-dir#gamma:0.001;T=0.46
    sub_runs_plots = filter(x -> x ∉ [265], range(1,no_runs,step=1))
    #320 run-dir#gamma:0.001;T=0.25
    sub_runs_plots = filter(x -> x ∉ [265], range(1,no_runs,step=1))

    for run_idx in sub_runs_plots
        γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
        Nx = N

        time_stamps = time_stamps
        dt = dt 

        # idxs=[(10+1),no_fft_snaps-max_spec_t_idx]
        idxs=[(10+1),(20+1),no_fft_snaps-max_spec_t_idx]
        idxs=[(10+1),(14+1)]
        
        #320#
        # idxs=[(10+1),(20+1),(30+1)]
        
        time_plot_idxs = idxs 
        # idxs=[(no_fft_snaps+1)÷2,no_fft_snaps-max_spec_t_idx]
        iter=1
        # for idx in range(2,size(restacked_fft,1),step=1)
        spec_stack_t = []

        # println(time_plot_idxs)
        # println(time_plot_idxs*dsnaps_fft*dt*mH);exit()

        for idx in time_plot_idxs

            # k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(run_dirs[run_idx],idx,run_idx)

            # B_fft_re,B_fft_re_binned,H_fft_re,H_fft_re_binned = load_less_processed_fft_data(run_dirs[run_idx],idx,run_idx)
            B_fft_re,B_fft_re_binned,H_fft_re,H_fft_re_binned,hel_fft_re,hel_fft_re_binned = load_less_processed_fft_data(run_dirs[run_idx],idx,run_idx)
            
            k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        
            k_phys,Hk_phys,k_phys_binned,Hk_phys_binned,Hk_phys_binned_err=phys_conv(H_fft_re,H_fft_re_binned,dx,Nx)

            ##E_M spectrum plot##
            plot!(p1,k_phys_binned[findall(k_phys_binned.!=0)],Bk_phys_binned[findall(k_phys_binned.!=0)],msc=colors[iter],c=colors[iter],
            label="",lw=0.25,alpha=0.5,ls=:dot)
            push!(spec_stack_t,Bk_phys_binned[findall(k_phys_binned.!=0)])
            kplot = k_phys_binned[findall(k_phys_binned.!=0)]
            iter=iter+1
            if (any(isnan.(Bk_phys_binned[findall(k_phys_binned.!=0)])))
                println(string("broken run:",run_idx," at ",idx))
                exit()
            end
            max_idx=(findall(Bk_phys_binned.>maximum(Bk_phys_binned)/3))[end]
            if (max_idx==size(kplot,1))
                println(string("blown run:",run_idx," at ",idx))
                exit()
            end
        end
        push!(spec_stack,spec_stack_t)
        
    end

    ##ensemble averaged spectra##
    iter = 1

    min_k = 100.0
    min_B_k = 1.0
    min_H_k = 1.0
    max_B_k = 1.0
    max_H_k = 1.0
    max_pad_factor = 1.5

    vert_shift = 1.2

    for idx in time_plot_idxs

        Bk_phys_binned_mean = mean([spec_stack[i][iter] for i in range(1,size(spec_stack,1),step=1)])
        Bk_phys_binned_err = std([spec_stack[i][iter] for i in range(1,size(spec_stack,1),step=1)])/sqrt.(no_runs)

        kplot=kplot[findall(kplot.!=0)]
        Bk_phys_binned_mean = Bk_phys_binned_mean[findall(kplot.!=0)]
        Bk_phys_binned_err = Bk_phys_binned_err[findall(kplot.!=0)]

        plot!(p1,kplot,Bk_phys_binned_mean,yerr=Bk_phys_binned_err,msc=colors[iter],c=colors[iter],
        label=string(round(time_stamps[idx]*dt*mH,digits=0)),lw=1.5)

        if idx==time_plot_idxs[end]

            m(t, p) = p[1]*t.+p[2]

            ##low_k fit
            p0 = [4.0, 1.0]

            low_k_cut_idx = findall(Bk_phys_binned_mean.==maximum(Bk_phys_binned_mean))[1]
            fit = curve_fit(m, log.(kplot[1:low_k_cut_idx]), log.(Bk_phys_binned_mean[1:low_k_cut_idx]), p0)

            # plot!(p1,kplot[1:low_k_cut_idx],exp.(vert_shift.*m(log.(kplot[1:low_k_cut_idx]),fit.param)),
            # ls=:dashdot,c=:black,label=string(L"$\alpha_1$:",round(fit.param[1],digits=1)))
            # vline!(p1,[kplot[low_k_cut_idx]],c=:black,ls=:dash,label="")

            ##tent-power laws##
            plot!(p1,kplot[1:low_k_cut_idx],
            exp.(m(log.(kplot[1:low_k_cut_idx]),[3.0,0.9*fit.param[2]])),
            ls=:dashdot,c=:black,label="")#,label=string(L"$\alpha_1$:",round(fit.param[1],digits=1)))

            ##High k fit
            p0 = [-4.0, 1.0]

            high_k_cut_idx = findall(Bk_phys_binned_mean.==maximum(Bk_phys_binned_mean))[1]
            high_k_cut_upp_idx =(findall(Bk_phys_binned_mean.>maximum(Bk_phys_binned_mean)/3))[end]
            fit = curve_fit(m, log.(kplot[high_k_cut_idx:high_k_cut_upp_idx]), log.(Bk_phys_binned_mean[high_k_cut_idx:high_k_cut_upp_idx]), p0)

            # plot!(p1,kplot[high_k_cut_idx:high_k_cut_upp_idx],exp.(vert_shift.*m(log.(kplot[high_k_cut_idx:high_k_cut_upp_idx]),fit.param)),
            # ls=:dashdot,c=:black,label=string(L"$\alpha_2$:",round(fit.param[1],digits=1)))
            
            # vline!(p1,[kplot[high_k_cut_upp_idx]],c=:red,ls=:dash,label="")

            ##tent fit##
            plot!(p1,kplot[high_k_cut_idx+1:high_k_cut_upp_idx+1],
            exp.(m(log.(kplot[high_k_cut_idx+1:high_k_cut_upp_idx+1]),[-1.0,0.95*fit.param[2]])),
            ls=:dashdot,c=:black,label="")#,label=string(L"$\alpha_1$:",round(fit.param[1],digits=1)))


            ###Very high k fit###
            high_k_cut_upp_idx =(findall(Bk_phys_binned_mean.>maximum(Bk_phys_binned_mean)/10))[end]
            veryhigh_k_cut_upp_idx = (findall(Bk_phys_binned_mean.>maximum(Bk_phys_binned_mean/1e4)))[end]
            fit = curve_fit(m, log.(kplot[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]), log.(Bk_phys_binned_mean[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]), p0)

            # plot!(p1,kplot[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx],exp.(vert_shift.*m(log.(kplot[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]),fit.param)),
            # ls=:dashdot,c=:black,label=string(L"$\alpha_3$:",round(fit.param[1],digits=1)))
            
            # vline!(p1,[kplot[veryhigh_k_cut_upp_idx]],c=:blue,ls=:dash,label="")

            ##tent fit##
            plot!(p1,kplot[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx],
            exp.(m(log.(kplot[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]),[-5.0,0.8*fit.param[2]])),
            ls=:dashdot,c=:black,label="")#,label=string(L"$\alpha_1$:",round(fit.param[1],digits=1)))
            
        end
        iter = iter+1

        if minimum(kplot)<=min_k
            min_k=minimum(kplot)
        end

        if minimum(Bk_phys_binned_mean)<min_B_k
            min_B_k=minimum(Bk_phys_binned_mean)
        end

        if maximum(Bk_phys_binned_mean)*max_pad_factor>max_B_k
            max_B_k=maximum(Bk_phys_binned_mean)*max_pad_factor
        end

    end
    
    plot!(p1,xlims=[min_k/1.1,k_xlim_upper/1.5],ylims=[max_B_k/5e2,max_B_k],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    # plot!(p1,ylims=[min_B_k,max_B_k],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    plot!(p1,xscale=:log,yscale=:log)
    plot!(p1,dpi=600)
    plot!(p1,xlabel=L"p/\eta",ylabel=L"4\pi p^2|B(p)|^2/V")
    plot!(p1,minorgrid=true,framestyle=:box)
    # png(p1,string(string(master_dir,"/",dir_name(para_arr[1])),"/post-spec-t-combined-log-ensemble.png"))

    # png(p1,string(string(master_dir,"/",dir_name(para_arr[1])),"/post-spec-t-combined-log-ensemble.png"))
    savefig(p1,string(string(master_dir,"/",dir_name(para_arr[1])),"/post-spec-t-combined-log-ensemble.pdf"))
end

export hosking_plot_therm
function hosking_plot_therm(para_arr;no_runs=25)
    
    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:hawaii25)
    colors=palette(:seaborn_bright)

    p1=plot([],[],label="")
    p2=plot([],[],label="")
    # plot!(p1,[],[],label="non-binned",seriestype=:scatter,c=:black)
    # plot!(p1,[],[],label="binned",c=:black)
    # plot!(p1,[],[],label=L"$p_{mean}$",c=:black,ls=:dash)
    # plot!(p1,[],[],label=L"fit ranges",c=:black,ls=:dash)
    plot!(p1,[],[],label=L"$m_Ht$",c=:white)

    time_plot_idxs = []
    kplot = []
    spec_stack = []

    time_stamps = []
    dt = 0.0 
    #320#
    sub_runs_plots = filter(x -> x ∉ [8,22,23], range(1,no_runs,step=1))
    #256#
    sub_runs_plots = filter(x -> x ∉ [17,10,2,6,8,11,12,22,25], range(1,no_runs,step=1))
    #320#dx:0.1#gamma:0.0
    # sub_runs_plots = filter(x -> x ∉ [1,2,3,4,14,17,21,24], range(1,no_runs,step=1))
    #256 run-dir#gamma:0.01
    # sub_runs_plots = filter(x -> x ∉ [5], range(1,no_runs,step=1))
    #256 run-dir#gamma:0.001;T=0.46
    sub_runs_plots = filter(x -> x ∉ [265], range(1,no_runs,step=1))
    #320 run-dir#gamma:0.001;T=0.25
    sub_runs_plots = filter(x -> x ∉ [265], range(1,no_runs,step=1))

    for run_idx in sub_runs_plots
        γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
        Nx = N

        time_stamps = time_stamps
        dt = dt 

        # idxs=[(10+1),no_fft_snaps-max_spec_t_idx]
        idxs=[(10+1),(20+1),no_fft_snaps-max_spec_t_idx]
        
        #320#
        # idxs=[(10+1),(20+1),(30+1)]
        
        time_plot_idxs = idxs 
        # idxs=[(no_fft_snaps+1)÷2,no_fft_snaps-max_spec_t_idx]
        iter=1
        # for idx in range(2,size(restacked_fft,1),step=1)
        spec_stack_t = []

        # println(time_plot_idxs)
        # println(time_plot_idxs*dsnaps_fft*dt*mH);exit()

        for idx in time_plot_idxs

            # k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(run_dirs[run_idx],idx,run_idx)

            # B_fft_re,B_fft_re_binned,H_fft_re,H_fft_re_binned = load_less_processed_fft_data(run_dirs[run_idx],idx,run_idx)
            B_fft_re,B_fft_re_binned,H_fft_re,H_fft_re_binned,hel_fft_re,hel_fft_re_binned = load_less_processed_fft_data(run_dirs[run_idx],idx,run_idx)
            
            k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        
            Hk_phys,Hk_phys_binned,Hk_phys_binned_err=phys_conv_H(H_fft_re,H_fft_re_binned,dx,Nx)

            ##H_M spectrum plot##
            plot!(p1,k_phys_binned[findall(k_phys_binned.!=0)],Hk_phys_binned[findall(k_phys_binned.!=0)]*2/(pi*(N*dx)^3),msc=colors[iter],c=colors[iter],
            label="",lw=0.25,alpha=0.5,ls=:dot)
            push!(spec_stack_t,Hk_phys_binned[findall(k_phys_binned.!=0)]*2/(pi*(N*dx)^3))
            kplot = k_phys_binned[findall(k_phys_binned.!=0)]
            iter=iter+1
            if (any(isnan.(Hk_phys_binned[findall(k_phys_binned.!=0)])))
                println(string("broken run:",run_idx," at ",idx))
                exit()
            end
            max_idx=(findall(Hk_phys_binned.>maximum(Hk_phys_binned)/3))[end]
            if (max_idx==size(kplot,1))
                println(string("blown run:",run_idx," at ",idx))
                exit()
            end
        end
        push!(spec_stack,spec_stack_t)
        
    end

    ##ensemble averaged spectra##
    iter = 1

    min_k = 100.0
    min_B_k = 1.0
    min_H_k = 1.0
    max_B_k = 1.0
    max_H_k = 1.0
    max_pad_factor = 1.5

    vert_shift = 1.2

    for idx in time_plot_idxs

        Hk_phys_binned_mean = mean([spec_stack[i][iter] for i in range(1,size(spec_stack,1),step=1)])
        Hk_phys_binned_err = std([spec_stack[i][iter] for i in range(1,size(spec_stack,1),step=1)])/sqrt.(no_runs)

        kplot=kplot[findall(kplot.!=0)]
        Hk_phys_binned_mean = Hk_phys_binned_mean[findall(kplot.!=0)]
        Hk_phys_binned_err = Hk_phys_binned_err[findall(kplot.!=0)]

        plot!(p1,kplot,Hk_phys_binned_mean,yerr=Hk_phys_binned_err,msc=colors[iter],c=colors[iter],
        label=string(round(time_stamps[idx]*dt*mH,digits=0)),lw=1.5)

        iter = iter+1

        if minimum(kplot)<=min_k
            min_k=minimum(kplot)
        end

        if minimum(Hk_phys_binned_mean)<min_B_k
            min_B_k=minimum(Hk_phys_binned_mean)
        end

        if maximum(Hk_phys_binned_mean)*max_pad_factor>max_B_k
            max_B_k=maximum(Hk_phys_binned_mean)*max_pad_factor
        end

    end
    
    # plot!(p1,xlims=[min_k/1.1,k_xlim_upper/1.5],ylims=[max_B_k/5e2,max_B_k],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    plot!(p1,xlims=[min_k/1.1,k_xlim_upper/1.5],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)

    # plot!(p1,ylims=[min_B_k,max_B_k],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    plot!(p1,xscale=:log,yscale=:log)
    plot!(p1,dpi=600)
    plot!(p1,xlabel=L"p/\eta",ylabel=L"h^2|B(p)|^2/(\pi V)")
    plot!(p1,minorgrid=true,framestyle=:box)
    # png(p1,string(string(master_dir,"/",dir_name(para_arr[1])),"/post-spec-t-combined-log-ensemble.png"))

    # png(p1,string(string(master_dir,"/",dir_name(para_arr[1])),"/post-spec-t-combined-log-ensemble.png"))
    savefig(p1,string(string(master_dir,"/",dir_name(para_arr[1])),"/post-hosk-t-combined-log-ensemble.pdf"))
end


export fft_plots_log_damp
function fft_plots_log_damp(para_arr)
    
    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:hawaii25)
    colors=palette(:seaborn_bright)

    p1=plot([],[],label="")
    p2=plot([],[],label="")
    # plot!(p1,[],[],label="non-binned",seriestype=:scatter,c=:black)
    # plot!(p1,[],[],label="binned",c=:black)
    # plot!(p1,[],[],label=L"$p_{mean}$",c=:black,ls=:dash)
    # plot!(p1,[],[],label=L"fit ranges",c=:black,ls=:dash)
    plot!(p1,[],[],label=L"$m_Ht$",c=:white)

    time_plot_idxs = []
    kplot = []
    spec_stack = []

    time_stamps = []
    dt = 0.0 

    for run_idx in range(1,size(para_arr,1),step=1)
        γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
        Nx = N

        time_stamps = time_stamps
        dt = dt 

        max_spec_t_idx = 0

        idxs=[no_fft_snaps-max_spec_t_idx]

        
        time_plot_idxs = idxs 
        iter=1
        spec_stack_t = []

        for idx in time_plot_idxs

            B_fft_re,B_fft_re_binned,H_fft_re,H_fft_re_binned,hel_fft_re,hel_fft_re_binned = load_less_processed_fft_data(run_dirs[run_idx],idx,run_idx)
            
            k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        
            k_phys,Hk_phys,k_phys_binned,Hk_phys_binned,Hk_phys_binned_err=phys_conv(H_fft_re,H_fft_re_binned,dx,Nx)

            ##E_M spectrum plot##
            plot!(p1,k_phys_binned[findall(k_phys_binned.!=0)],Bk_phys_binned[findall(k_phys_binned.!=0)],msc=colors[iter],c=colors[iter],
            label="",lw=0.25,alpha=0.5,ls=:dot)
            push!(spec_stack_t,Bk_phys_binned[findall(k_phys_binned.!=0)])
            kplot = k_phys_binned[findall(k_phys_binned.!=0)]
            iter=iter+1
            if (any(isnan.(Bk_phys_binned[findall(k_phys_binned.!=0)])))
                println(string("broken run:",run_idx," at ",idx))
                exit()
            end
            max_idx=(findall(Bk_phys_binned.>maximum(Bk_phys_binned)/3))[end]
            if (max_idx==size(kplot,1))
                println(string("blown run:",run_idx," at ",idx))
                exit()
            end
        end
        push!(spec_stack,spec_stack_t)
        
    end

    ##ensemble averaged spectra##
    iter = 1

    min_k = 100.0
    min_B_k = 1.0
    min_H_k = 1.0
    max_B_k = 1.0
    max_H_k = 1.0
    max_pad_factor = 1.5

    vert_shift = 1.2

    for idx in time_plot_idxs

        Bk_phys_binned_mean = mean([spec_stack[i][iter] for i in range(1,size(spec_stack,1),step=1)])
        Bk_phys_binned_err = std([spec_stack[i][iter] for i in range(1,size(spec_stack,1),step=1)])/sqrt.(no_runs)

        kplot=kplot[findall(kplot.!=0)]
        Bk_phys_binned_mean = Bk_phys_binned_mean[findall(kplot.!=0)]
        Bk_phys_binned_err = Bk_phys_binned_err[findall(kplot.!=0)]

        plot!(p1,kplot,Bk_phys_binned_mean,yerr=Bk_phys_binned_err,msc=colors[iter],c=colors[iter],
        label=string(round(time_stamps[idx]*dt*mH,digits=0)),lw=1.5)

        if idx==time_plot_idxs[end]

            m(t, p) = p[1]*t.+p[2]

            ##low_k fit
            p0 = [4.0, 1.0]

            low_k_cut_idx = findall(Bk_phys_binned_mean.==maximum(Bk_phys_binned_mean))[1]
            fit = curve_fit(m, log.(kplot[1:low_k_cut_idx]), log.(Bk_phys_binned_mean[1:low_k_cut_idx]), p0)

            # plot!(p1,kplot[1:low_k_cut_idx],exp.(vert_shift.*m(log.(kplot[1:low_k_cut_idx]),fit.param)),
            # ls=:dashdot,c=:black,label=string(L"$\alpha_1$:",round(fit.param[1],digits=1)))
            # vline!(p1,[kplot[low_k_cut_idx]],c=:black,ls=:dash,label="")

            ##tent-power laws##
            plot!(p1,kplot[1:low_k_cut_idx],
            exp.(m(log.(kplot[1:low_k_cut_idx]),[3.0,0.9*fit.param[2]])),
            ls=:dashdot,c=:black,label="")#,label=string(L"$\alpha_1$:",round(fit.param[1],digits=1)))

            ##High k fit
            p0 = [-4.0, 1.0]

            high_k_cut_idx = findall(Bk_phys_binned_mean.==maximum(Bk_phys_binned_mean))[1]
            high_k_cut_upp_idx =(findall(Bk_phys_binned_mean.>maximum(Bk_phys_binned_mean)/3))[end]
            fit = curve_fit(m, log.(kplot[high_k_cut_idx:high_k_cut_upp_idx]), log.(Bk_phys_binned_mean[high_k_cut_idx:high_k_cut_upp_idx]), p0)

            # plot!(p1,kplot[high_k_cut_idx:high_k_cut_upp_idx],exp.(vert_shift.*m(log.(kplot[high_k_cut_idx:high_k_cut_upp_idx]),fit.param)),
            # ls=:dashdot,c=:black,label=string(L"$\alpha_2$:",round(fit.param[1],digits=1)))
            
            # vline!(p1,[kplot[high_k_cut_upp_idx]],c=:red,ls=:dash,label="")

            ##tent fit##
            plot!(p1,kplot[high_k_cut_idx+1:high_k_cut_upp_idx+1],
            exp.(m(log.(kplot[high_k_cut_idx+1:high_k_cut_upp_idx+1]),[-1.0,0.95*fit.param[2]])),
            ls=:dashdot,c=:black,label="")#,label=string(L"$\alpha_1$:",round(fit.param[1],digits=1)))


            ###Very high k fit###
            high_k_cut_upp_idx =(findall(Bk_phys_binned_mean.>maximum(Bk_phys_binned_mean)/10))[end]
            veryhigh_k_cut_upp_idx = (findall(Bk_phys_binned_mean.>maximum(Bk_phys_binned_mean/1e4)))[end]
            fit = curve_fit(m, log.(kplot[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]), log.(Bk_phys_binned_mean[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]), p0)

            # plot!(p1,kplot[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx],exp.(vert_shift.*m(log.(kplot[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]),fit.param)),
            # ls=:dashdot,c=:black,label=string(L"$\alpha_3$:",round(fit.param[1],digits=1)))
            
            # vline!(p1,[kplot[veryhigh_k_cut_upp_idx]],c=:blue,ls=:dash,label="")

            ##tent fit##
            plot!(p1,kplot[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx],
            exp.(m(log.(kplot[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]),[-5.0,0.8*fit.param[2]])),
            ls=:dashdot,c=:black,label="")#,label=string(L"$\alpha_1$:",round(fit.param[1],digits=1)))
            
        end
        iter = iter+1

        if minimum(kplot)<=min_k
            min_k=minimum(kplot)
        end

        if minimum(Bk_phys_binned_mean)<min_B_k
            min_B_k=minimum(Bk_phys_binned_mean)
        end

        if maximum(Bk_phys_binned_mean)*max_pad_factor>max_B_k
            max_B_k=maximum(Bk_phys_binned_mean)*max_pad_factor
        end

    end
    
    plot!(p1,xlims=[min_k/1.1,k_xlim_upper/1.5],ylims=[max_B_k/5e2,max_B_k],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    plot!(p1,xscale=:log,yscale=:log)
    plot!(p1,dpi=600)
    plot!(p1,xlabel=L"p/\eta",ylabel=L"4\pi p^2|B(p)|^2/V")
    plot!(p1,minorgrid=true,framestyle=:box)
    savefig(p1,string(master_dir,"/gamma-plots","/log-spec-damp-compare.png"))
end

export fft_plots_log_therm_early
function fft_plots_log_therm_early(para_arr;no_runs=25)
    
    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:hawaii25)
    colors=palette(:seaborn_bright)

    p1=plot([],[],label="")
    p2=plot([],[],label="")
    # plot!(p1,[],[],label="non-binned",seriestype=:scatter,c=:black)
    # plot!(p1,[],[],label="binned",c=:black)
    # plot!(p1,[],[],label=L"$p_{mean}$",c=:black,ls=:dash)
    plot!(p1,[],[],label=L"fit ranges",c=:black,ls=:dash)
    plot!(p1,[],[],label=L"$m_Ht$",c=:white)

    time_plot_idxs = []
    kplot = []
    spec_stack = []

    time_stamps = []
    dt = 0.0 
    sub_runs_plots = filter(x -> x ∉ [8,22,23], range(1,no_runs,step=1))

    for run_idx in sub_runs_plots
        γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
        Nx = N

        time_stamps = time_stamps
        dt = dt 

        idxs=[(10+1),no_fft_snaps-max_spec_t_idx]
        # idxs=[(10+1),(20+1),no_fft_snaps-max_spec_t_idx]
        idxs=[(10+1),(20+1),(30+1)]
        idxs=[1,2,25]

        time_plot_idxs = idxs 
        # idxs=[(no_fft_snaps+1)÷2,no_fft_snaps-max_spec_t_idx]
        iter=1
        # for idx in range(2,size(restacked_fft,1),step=1)
        spec_stack_t = []

        # println(time_plot_idxs)
        # println(time_plot_idxs*dsnaps_fft*dt*mH);exit()

        for idx in time_plot_idxs

            # k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(run_dirs[run_idx],idx,run_idx)

            B_fft_re,B_fft_re_binned,H_fft_re,H_fft_re_binned = load_less_processed_fft_data(run_dirs[run_idx],idx,run_idx)

            k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        
            k_phys,Hk_phys,k_phys_binned,Hk_phys_binned,Hk_phys_binned_err=phys_conv(H_fft_re,H_fft_re_binned,dx,Nx)

            ##E_M spectrum plot##
            plot!(p1,k_phys_binned[findall(k_phys_binned.!=0)],Bk_phys_binned[findall(k_phys_binned.!=0)],msc=colors[iter],c=colors[iter],
            label="",lw=0.25,alpha=0.5,ls=:dot)
            push!(spec_stack_t,Bk_phys_binned[findall(k_phys_binned.!=0)])
            kplot = k_phys_binned[findall(k_phys_binned.!=0)]
            iter=iter+1

        end
        push!(spec_stack,spec_stack_t)
        
    end

    ##ensemble averaged spectra##
    iter = 1

    min_k = 100.0
    min_B_k = 1.0
    min_H_k = 1.0
    max_B_k = 1.0
    max_H_k = 1.0
    max_pad_factor = 1.5

    for idx in time_plot_idxs

        Bk_phys_binned_mean = mean([spec_stack[i][iter] for i in range(1,size(spec_stack,1),step=1)])
        Bk_phys_binned_err = std([spec_stack[i][iter] for i in range(1,size(spec_stack,1),step=1)])/sqrt.(no_runs)

        kplot=kplot[findall(kplot.!=0)]
        Bk_phys_binned_mean = Bk_phys_binned_mean[findall(kplot.!=0)]
        Bk_phys_binned_err = Bk_phys_binned_err[findall(kplot.!=0)]

        plot!(p1,kplot,Bk_phys_binned_mean,yerr=Bk_phys_binned_err,msc=colors[iter],c=colors[iter],
        label=string(round(time_stamps[idx]*dt*mH,digits=0)),lw=1.5)

        m(t, p) = p[1]*t.+p[2]

        ##low_k fit
        p0 = [4.0, 1.0]

        low_k_cut_idx = findall(Bk_phys_binned_mean.==maximum(Bk_phys_binned_mean))[1]
        low_k_cut_idx = 3
        fit = curve_fit(m, log.(kplot[1:low_k_cut_idx]), log.(Bk_phys_binned_mean[1:low_k_cut_idx]), p0)
        plot!(p1,kplot[1:low_k_cut_idx],exp.(m(log.(kplot[1:low_k_cut_idx]),fit.param)),
        ls=:dashdot,c=colors[iter],label=string(L"$\alpha_1$:",round(fit.param[1],digits=1)))
        vline!(p1,[kplot[low_k_cut_idx]],c=:black,ls=:dash,label="")

        ##High k fit
        p0 = [-4.0, 1.0]

        high_k_cut_idx = findall(Bk_phys_binned_mean.==maximum(Bk_phys_binned_mean))[1]
        high_k_cut_upp_idx =(findall(Bk_phys_binned_mean.>maximum(Bk_phys_binned_mean)/3))[end]

        fit = curve_fit(m, log.(kplot[high_k_cut_idx:high_k_cut_upp_idx]), log.(Bk_phys_binned_mean[high_k_cut_idx:high_k_cut_upp_idx]), p0)
        plot!(p1,kplot[high_k_cut_idx:high_k_cut_upp_idx],exp.(m(log.(kplot[high_k_cut_idx:high_k_cut_upp_idx]),fit.param)),
        ls=:dashdot,c=colors[iter],label=string(L"$\alpha_2$:",round(fit.param[1],digits=1)))
        
        vline!(p1,[kplot[high_k_cut_upp_idx]],c=:red,ls=:dash,label="")

        ###Very high k fit###
        veryhigh_k_cut_upp_idx = (findall(Bk_phys_binned_mean.>maximum(Bk_phys_binned_mean/1e4)))[end]
        fit = curve_fit(m, log.(kplot[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]), log.(Bk_phys_binned_mean[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]), p0)
        plot!(p1,kplot[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx],exp.(m(log.(kplot[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]),fit.param)),
        ls=:dashdot,c=colors[iter],label=string(L"$\alpha_3$:",round(fit.param[1],digits=1)))
        
        vline!(p1,[kplot[veryhigh_k_cut_upp_idx]],c=:blue,ls=:dash,label="")

        iter = iter+1


        if minimum(kplot)<=min_k
            min_k=minimum(kplot)
        end

        if minimum(Bk_phys_binned_mean)<min_B_k
            min_B_k=minimum(Bk_phys_binned_mean)
        end

        if maximum(Bk_phys_binned_mean)*max_pad_factor>max_B_k
            max_B_k=maximum(Bk_phys_binned_mean)*max_pad_factor
        end

    end
    
    plot!(p1,xlims=[min_k/2.,k_xlim_upper],ylims=[max_B_k/10000,max_B_k],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    plot!(p1,xscale=:log,yscale=:log)
    plot!(p1,dpi=600)
    plot!(p1,xlabel=L"p/\eta",ylabel=L"4\pi p^2|B(p)|^2/V")
    plot!(p1,minorgrid=true,framestyle=:box)
    # png(p1,string(string(master_dir,"/",dir_name(para_arr[1])),"/post-spec-t-combined-log-ensemble.png"))

    png(p1,string(string(master_dir,"/",dir_name(para_arr[1])),"/post-spec-t-combined-log-ensemble-earlytime.png"))

end

export peak_evo_plot
function peak_evo_plot(para_arr;run_idx=1)
    γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
    Nx = N

    time_start = 1
    peaks_arr = zeros((no_fft_snaps-time_start,6))
    for idx in range(time_start,no_fft_snaps-max_spec_t_idx,step=1)

        k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(run_dirs[run_idx],idx,run_idx)
        k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        
        k_phys,Hk_phys,k_phys_binned,Hk_phys_binned,Hk_phys_binned_err=phys_conv(H_fft_re,H_fft_re_binned,dx,Nx)

        # peaks_arr = find_mean_peaks(restacked_fft)
        t,k_mean_nonbinned,k_mean_binned,k_max,xi_mean_nonbinned,xi_mean_binned = find_mean_peaks(k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,idx,run_idx)
        peaks_arr[idx,:] = [t,k_mean_nonbinned,k_mean_binned,k_max,xi_mean_nonbinned,xi_mean_binned]
        println("t: ", t," xi: ",xi_mean_binned)
    end
    # exit()
    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:tab10)
    p1=plot([],[],label="")
    

    # plot!(p1,peaks_arr[:,1],2*pi./peaks_arr[:,2],label="Non-Binned")
    # plot!(p1,peaks_arr[:,1],2*pi./peaks_arr[:,3],label="Binned")
    plot!(p1,peaks_arr[:,1],2*pi./peaks_arr[:,4],shape=:circle,ms=2.0,ls=:dot,label=L"p_{max}")
    plot!(p1,peaks_arr[:,1],peaks_arr[:,5],label="non-binned",shape=:circle,ms=2.0,ls=:dash)
    plot!(p1,peaks_arr[:,1],peaks_arr[:,6]
    ,shape=:circle,ms=2.0,label="binned")
    
    plot!(p1,xlabel=L"m_Ht",ylabel=L"\xi_M")
    plot!(p1,xscale=:log,yscale=:log)
    plot!(p1,xscale=:linear,yscale=:linear)
    
    plot!(p1,minorgrid=true,framestyle=:box)
    plot!(p1,dpi=600)
    
    png(p1,string(run_dirs[run_idx],"/peak-scale-evo.png"))

    return
end


export peak_evo_size_plot
function peak_evo_size_plot(para_arr)

    time_start = 1
    peaks_arr = zeros((size(para_arr,1),no_fft_snaps-time_start,6))

    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:tab10)
    p1=plot([],[],label="")

    for run_idx in range(1,size(para_arr,1),step=1)

        γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
        Nx = N

        for idx in range(time_start,no_fft_snaps,step=5)

            k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(run_dirs[run_idx],idx,run_idx)
            k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        
            k_phys,Hk_phys,k_phys_binned,Hk_phys_binned,Hk_phys_binned_err=phys_conv(H_fft_re,H_fft_re_binned,dx,Nx)

            # peaks_arr = find_mean_peaks(restacked_fft)
            t,k_mean_nonbinned,k_mean_binned,k_max,xi_mean_nonbinned,xi_mean_binned = find_mean_peaks(k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,idx,run_idx)
            peaks_arr[idx,:] = [t,k_mean_nonbinned,k_mean_binned,k_max,xi_mean_nonbinned,xi_mean_binned]
            println("t: ", t," xi: ",xi_mean_binned)
        end
        # exit()
        plot!(p1,peaks_arr[:,1],peaks_arr[:,6],shape=:circle,ms=2.0,label="binned")
    
    end
    
    plot!(p1,xlabel=L"m_Ht",ylabel=L"\xi_M")
    # plot!(p1,xscale=:log,yscale=:log)
    plot!(p1,xscale=:linear,yscale=:linear)
    
    plot!(p1,minorgrid=true,framestyle=:box)
    plot!(p1,dpi=600)
    
    png(p1,string(run_dirs[run_idx],"/peak-scale-evo.png"))

    return
end

export peak_evo__therm
function peak_evo__therm(para_arr;no_runs=25)
    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:tab10)
    p1=plot([],[],label="")

    time_stamps = []
    peaks_arr = []
    t =[]
    dt = 0.0
    L_box = 0.0
    #select good runs ffor 320 runs#
    sub_runs_plots = filter(x -> x ∉ [8,22,23], range(1,no_runs,step=1))
    #select good runs for N=256#
    sub_runs_plots = filter(x -> x ∉ [17], range(1,no_runs,step=1))
    #256#
    sub_runs_plots = filter(x -> x ∉ [17,10,2,6,8,11,12,22,25], range(1,no_runs,step=1))
    #320#dx:0.1#gamma:0.0
    sub_runs_plots = filter(x -> x ∉ [26], range(1,no_runs,step=1))
    #256 run-dir#gamma:0.01
    sub_runs_plots = filter(x -> x ∉ [5], range(1,no_runs,step=1))

    for run_idx in sub_runs_plots
        γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
        Nx = N
        L_box = N*dx
        time_stamps = time_stamps
        dt = dt 
        run_stack = []
        t_arr = []
        time_start = 1
        # println(run_idx)
        plot_time_idxs = range(time_start,no_fft_snaps-max_spec_t_idx,step=1)
        # plot_time_idxs=filter(x -> x ∉ [21], plot_time_idxs)
        for idx in plot_time_idxs
            println(idx)
            # k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(run_dirs[run_idx],idx,run_idx)
    
            B_fft_re,B_fft_re_binned,H_fft_re,H_fft_re_binned = load_less_processed_fft_data(run_dirs[run_idx],idx,run_idx)
            k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        
            k_phys,Hk_phys,k_phys_binned,Hk_phys_binned,Hk_phys_binned_err=phys_conv(H_fft_re,H_fft_re_binned,dx,Nx)

            # peaks_arr = find_mean_peaks(restacked_fft)
            # t,k_mean_nonbinned,k_mean_binned,k_max,xi_mean_nonbinned,xi_mean_binned = find_mean_peaks_less(k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,idx,run_idx)
            t,k_mean_nonbinned,k_mean_binned,k_max,xi_mean_nonbinned,xi_mean_binned = find_mean_peaks_less(B_fft_re,B_fft_re_binned,idx,run_idx)
            push!(run_stack,xi_mean_binned)
            push!(t_arr,time_stamps[idx]*dt*mH)
            
        end
        
        if size(run_stack,1)==size(plot_time_idxs,1)
            println(string(run_idx,", ", size(run_stack,1), " ", size(plot_time_idxs,1)," ", size(run_stack)))
            push!(peaks_arr,run_stack)
        end
        
        t=t_arr

        plot!(t,run_stack./L_box,lw=0.4,alpha=0.5,ls=:dot,label="")
        # plot!(t,run_stack,ls=:dot,label=run_idx)

    end
    println(size(peaks_arr))
    len_xi = size(peaks_arr[1],1)
    xi_mean = [mean([peaks_arr[i][iter] for i in range(1,size(peaks_arr,1),step=1)]) for iter in range(1,len_xi,step=1)]
    # println(xi_mean);exit()
    plot!(p1,t,xi_mean./L_box,label="")
  
    plot!(p1,xscale=:linear,yscale=:linear)
    plot!(p1,dpi=600)
    plot!(p1,minorgrid=true,framestyle=:box)
    png(p1,string(string(master_dir,"/",dir_name(para_arr[1])),"/post-evo-plots-ensemble.png"))

end

export peak_evo__therm_ens_compare
function peak_evo__therm_ens_compare(para_arr;ratio=false,no_runs=25,N_arr=[256,320],rewrite=false)

    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:tab10)
    p1=plot([],[],label="")
    plot!(p1,[],[],label=L"$N$",c=:white)

    bad_256_idxs = [17,10,2,6,8,11,12,22,25]
    bad_256_idxs = [17]
    bad_320_idxs = [8,22,23]
    bad_idx_stack = [bad_256_idxs,bad_320_idxs]

    for (N_idx,N_i) in enumerate(N_arr)
        println(N_i)
        time_stamps = []
        peaks_arr = []
        t =[]
        dt = 0.0
        L_box = 0.0

        sub_runs_plots = filter(x -> x ∉ bad_idx_stack[N_idx], range(1,no_runs,step=1))

        for run_idx in sub_runs_plots
            γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx+(N_idx-1)*25],para_arr[run_idx+(N_idx-1)*25])
            Nx = N
            
            if ratio==true
                L_box = N*dx
            else
                L_box = 1   #forr plotting x_m not the ratio
            end

            dt = dt 
            run_stack = []
            t_arr = []
            time_start = 1
            # println(run_idx)
            plot_time_idxs = range(time_start,no_fft_snaps-max_spec_t_idx,step=1)
            # plot_time_idxs=filter(x -> x ∉ [21], plot_time_idxs)

            if ((N==N_i))

                for idx in plot_time_idxs
                    # println(idx)
                    # k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(run_dirs[run_idx],idx,run_idx)
            
                    B_fft_re,B_fft_re_binned,H_fft_re,H_fft_re_binned = load_less_processed_fft_data(run_dirs[run_idx+(N_idx-1)*25],idx,run_idx)
                    k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        
                    k_phys,Hk_phys,k_phys_binned,Hk_phys_binned,Hk_phys_binned_err=phys_conv(H_fft_re,H_fft_re_binned,dx,Nx)

                    # peaks_arr = find_mean_peaks(restacked_fft)
                    # t,k_mean_nonbinned,k_mean_binned,k_max,xi_mean_nonbinned,xi_mean_binned = find_mean_peaks_less(k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,idx,run_idx)
                    t,k_mean_nonbinned,k_mean_binned,k_max,xi_mean_nonbinned,xi_mean_binned = find_mean_peaks_less(B_fft_re,B_fft_re_binned,idx,run_idx+(N_idx-1)*25)
                    push!(run_stack,xi_mean_binned)
                    push!(t_arr,time_stamps[idx]*dt)
                    
                end
                
                if size(run_stack,1)==size(plot_time_idxs,1)
                    println(string(run_idx,", ", size(run_stack,1), " ", size(plot_time_idxs,1)," ", size(run_stack)))
                    push!(peaks_arr,run_stack)
                end
                
                t=t_arr

                plot!(t,run_stack./L_box,lw=0.4,alpha=0.5,ls=:dot,label="",color=colors[N_idx])
                # plot!(t,run_stack,ls=:dot,label=run_idx)
            end
        end
        println(size(peaks_arr))
        len_xi = size(peaks_arr[1],1)
        xi_mean = [mean([peaks_arr[i][iter] for i in range(1,size(peaks_arr,1),step=1)]) for iter in range(1,len_xi,step=1)]
        xi_std = [std([peaks_arr[i][iter] for i in range(1,size(peaks_arr,1),step=1)]./L_box)./sqrt(size(peaks_arr,1)) for iter in range(1,len_xi,step=1)]
        # println(xi_mean);exit()
        plot!(p1,t,xi_mean./L_box,yerr=xi_std,label=N_i,color=colors[N_idx])
    end

    plot!(p1,xscale=:linear,yscale=:linear)
    if ratio==true
        plot!(p1,xlabel=L"\eta t",ylabel = L"\xi_M/L ")
    else
        plot!(p1,xlabel=L"\eta t",ylabel = L"\eta\xi_M")
    end

    plot!(p1,dpi=600)
    plot!(p1,minorgrid=true,framestyle=:box)
    if ratio==true
        png(p1,string(master_dir,"/post-evo-plots-ensemble-compare-ratio.png"))
    else
        png(p1,string(master_dir,"/post-evo-plots-ensemble-compare.png"))
    end

end

export prep_therm
function prep_therm(para_arr;run_idx=1,idx)
    γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
    time_start = 1

    # Threads.@threads for idx in range(time_start,no_fft_snaps-max_spec_t_idx,step=1)
    # for idx in range(time_start,no_fft_snaps-max_spec_t_idx,step=1)
        # println(idx)
        k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(run_dirs[run_idx],idx,run_idx)
        k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        
        k_phys,Hk_phys,k_phys_binned,Hk_phys_binned,Hk_phys_binned_err=phys_conv(H_fft_re,H_fft_re_binned,dx,Nx)

        # t,k_mean_nonbinned,k_mean_binned,k_max,xi_mean_nonbinned,xi_mean_binned = find_mean_peaks(k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,idx,run_idx)
        # println("t: ", t," xi: ",xi_mean_binned)
    # end
end

export prep_therm_all
function prep_therm_all(para_arr;run_idx=1)
    γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
    time_start = 1

    # Threads.@threads for idx in range(time_start,no_fft_snaps-max_spec_t_idx,step=1)
    for idx in range(time_start,no_fft_snaps-max_spec_t_idx,step=1)
        # println(idx)
        k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(run_dirs[run_idx],idx,run_idx)
        k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        
        k_phys,Hk_phys,k_phys_binned,Hk_phys_binned,Hk_phys_binned_err=phys_conv(H_fft_re,H_fft_re_binned,dx,Nx)

        # t,k_mean_nonbinned,k_mean_binned,k_max,xi_mean_nonbinned,xi_mean_binned = find_mean_peaks(k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,idx,run_idx)
        # println("t: ", t," xi: ",xi_mean_binned)
    end
end

export energy_plots_therm
function energy_plots_therm(para_arr;no_runs=25)
    
    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:tab10)
    p1=plot([],[],label="")
    p2=plot([],[],label="")
    p3=plot([],[],label="")
    p4=plot([],[],label="")
    p5=plot([],[],label="")
    p6=plot([],[],label="")
    p7=plot([],[],label="")

    # #select good runs for 320 runs#
    sub_runs_plots = filter(x -> x ∉ [8,22,23], range(1,no_runs,step=1))
    # #select good runs for N=256#
    # sub_runs_plots = filter(x -> x ∉ [60], range(1,no_runs,step=1))

    for run_idx in sub_runs_plots
        γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
        Nx = N
        
        run_dir = run_dirs[run_idx]
        total_energies= load_data(run_dir)

        box_vol = (Nx*dx)^3

        println(size(total_energies))#;exit()

        tot_energy_snp = [sum(total_energies[a,1:end-6]) for a in range(1,nsnaps+1,step=1)]./box_vol
        total_gauge_e = [sum(total_energies[a,4:7]) for a in range(1,nsnaps+1,step=1)]./box_vol
        total_phi_pot = total_energies[:,1]./box_vol
        total_phi_ke = total_energies[:,2]./box_vol
        total_phi_ge = total_energies[:,3]./box_vol
        total_w_ee = total_energies[:,4]./box_vol
        total_w_me = total_energies[:,5]./box_vol
        total_y_ee = total_energies[:,6]./box_vol
        total_y_me = total_energies[:,7]./box_vol
        total_b_eme = total_energies[:,8]./box_vol
        total_b2_eme = total_energies[:,9]./box_vol
        total_b3_eme = total_energies[:,11]./box_vol
        avgphi = total_energies[:,12]
        minphi = total_energies[:,10]

        # plot!(p1,range(0,nte,step=dsnaps).*dt*mH,(total_phi_pot.+total_phi_ke.+total_phi_ge)./tot_energy_snp[1],
        # label=run_idx)
        plot!(p1,range(0,nte,step=dsnaps).*dt*mH,tot_energy_snp./tot_energy_snp[1],
        label=run_idx)

        plot!(p2,range(0,nte,step=dsnaps).*dt*mH,minphi,xlabel=L"m_Ht",ls=:dashdot,label="")
        plot!(p2,range(0,nte,step=dsnaps).*dt*mH,avgphi,label=run_idx)

        t = range(0,nte,step=dsnaps).*dt*mH
        pe=(total_phi_pot.-lambda*vev^4).*(mH)^4
        ke=(total_phi_ke).*(mH)^4
        ge=(total_phi_ge).*(mH)^4
        me=(total_b_eme).*(mH)^4
        me2=(total_b2_eme).*(mH)^4
        total_energy = abs.(ke+ge+pe).-abs.(pe[1].+ke[1].+ge[1])
        plot!(p3,t,me./tot_energy_snp[1],label=run_idx)

        plot!(p4,range(0,nte,step=dsnaps).*dt*mH,minphi,xlabel=L"m_Ht",label=run_idx)
        # ptwin =twiny()
        # plot!(ptwin,range(0,nte,step=dsnaps).*dt/(Nx*dx),minphi,xlabel=L"t/T_{lc}")
    end

    # plot!(p1,framestyle=:box,ylims=[0.0,1.1])
    plot!(p1,framestyle=:box,ylims=[0.9,1.1])

    plot!(p1,ylabel=L"E(t)/E(0)")

    plot!(p2,framestyle=:box,ylims=[0.0,1.5])
    hline!(p2,[1,1],ls=:dot,color=:black,label="")

    plot(p1,p2,layout=grid(2, 1, heights=[0.6, 0.4]),dpi=600)
    png(string(string(master_dir,"/",dir_name(para_arr[1])), "/ens-avg-higgs-evo.png"))

    plot!(p3,xlabel=L"m_H t",ylabel=L"{E_B}/{E(0)}")
    plot!(p3,ylims=[1e-3,1e-1])
    plot!(p3,grid=true,gridlinewidth=2.0,yscale=:log,dpi=600)
    plot!(p3,dpi=600)
    png(p3,string(string(master_dir,"/",dir_name(para_arr[1])),"/ens-avg-mag-energy-evo.png"))


    plot!(p4,dpi=600)
    png(p4,string(string(master_dir,"/",dir_name(para_arr[1])),"/ens-avg-min-phi.png"))

    return
end


export energy_plots_therm_early
function energy_plots_therm_early(para_arr;no_runs=25)
    
    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:tab10)
    p1=plot([],[],label="")
    p2=plot([],[],label="")
    p3=plot([],[],label="")
    p4=plot([],[],label="")
    p5=plot([],[],label="")
    p6=plot([],[],label="")
    p7=plot([],[],label="")

    # #select good runs for 320 runs#
    sub_runs_plots = filter(x -> x ∉ [8,22,23], range(1,no_runs,step=1))
    # #select good runs for N=256#
    # sub_runs_plots = filter(x -> x ∉ [60], range(1,no_runs,step=1))

    stck_1 = []
    stck_2 = []
    stck_3 = []
    stck_4 = []
    stck_5 = []
    stck_6 = []
    stck_7 = []
    tstck = []
    for run_idx in sub_runs_plots
        γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
        Nx = N
        
        run_dir = run_dirs[run_idx]
        total_energies= load_data(run_dir)

        box_vol = (Nx*dx)^3

        println(size(total_energies))#;exit()

        tot_energy_snp = [sum(total_energies[a,1:end-6]) for a in range(1,nsnaps+1,step=1)]./box_vol
        total_gauge_e = [sum(total_energies[a,4:7]) for a in range(1,nsnaps+1,step=1)]./box_vol
        total_phi_pot = total_energies[:,1]./box_vol
        total_phi_ke = total_energies[:,2]./box_vol
        total_phi_ge = total_energies[:,3]./box_vol
        total_w_ee = total_energies[:,4]./box_vol
        total_w_me = total_energies[:,5]./box_vol
        total_y_ee = total_energies[:,6]./box_vol
        total_y_me = total_energies[:,7]./box_vol
        total_b_eme = total_energies[:,8]./box_vol
        total_b2_eme = total_energies[:,9]./box_vol
        total_b3_eme = total_energies[:,11]./box_vol
        avgphi = total_energies[:,12]
        minphi = total_energies[:,10]

        plot!(p1,range(0,nte,step=dsnaps).*dt*mH,tot_energy_snp./tot_energy_snp[1],
        label=run_idx)

        plot!(p2,range(0,nte,step=dsnaps).*dt*mH,minphi,xlabel=L"m_Ht",ls=:dashdot,label="")
        plot!(p2,range(0,nte,step=dsnaps).*dt*mH,avgphi,label=run_idx)

        t = range(0,nte,step=dsnaps).*dt*mH
        pe=(total_phi_pot.-lambda*vev^4).*(mH)^4
        ke=(total_phi_ke).*(mH)^4
        ge=(total_phi_ge).*(mH)^4
        me=(total_b_eme).*(mH)^4
        me2=(total_b2_eme).*(mH)^4
        total_energy = abs.(ke+ge+pe).-abs.(pe[1].+ke[1].+ge[1])

        push!(stck_1,(total_phi_ge)./tot_energy_snp[1])
        push!(stck_2,(total_phi_ke)./tot_energy_snp[1])

        plot!(p5,t,(total_phi_ge)./tot_energy_snp[1],lw=0.5,alpha=0.5,c=colors[1],label="",ls=:dash)
        plot!(p5,t,(total_phi_ke)./tot_energy_snp[1],lw=0.5,alpha=0.5,c=colors[2],label="",ls=:dash)

        push!(stck_3,(total_w_me)./tot_energy_snp[1])
        push!(stck_4,(total_w_ee)./tot_energy_snp[1])

        plot!(p6,t,(total_w_me)./tot_energy_snp[1],lw=0.5,alpha=0.5,c=colors[1],label="",ls=:dash)
        plot!(p6,t,(total_w_ee)./tot_energy_snp[1],lw=0.5,alpha=0.5,c=colors[2],label="",ls=:dash)

        push!(stck_5,(total_y_me)./tot_energy_snp[1])
        push!(stck_6,(total_y_ee)./tot_energy_snp[1])

        plot!(p7,t,(total_y_me)./tot_energy_snp[1],lw=0.5,alpha=0.5,c=colors[1],label="",ls=:dash)
        plot!(p7,t,(total_y_ee)./tot_energy_snp[1],lw=0.5,alpha=0.5,c=colors[2],label="",ls=:dash)

        tstck = t
    end

    # plot!(p1,framestyle=:box,ylims=[0.0,1.1])
    # plot!(p1,framestyle=:box,ylims=[0.9,1.1])

    # plot!(p1,ylabel=L"E(t)/E(0)")

    # plot!(p2,framestyle=:box,ylims=[0.0,1.5])
    # hline!(p2,[1,1],ls=:dot,color=:black,label="")

    # plot(p1,p2,layout=grid(2, 1, heights1,dir_name(para_arr[1])),"/ens-avg-mag-energy-evo.png"))


    # plot!(p4,dpi=600)
    # png(p4,string(string(master_dir,"/",dir_name(para_arr[1])),"/ens-avg-min-phi.png"))

    avg_phi_ge = mean(stck_1)
    avg_phi_ke = mean(stck_2)
    avg_w_me = mean(stck_3)
    avg_w_ee = mean(stck_4)
    avg_y_me = mean(stck_5)
    avg_y_ee = mean(stck_6)

    plot!(p5,tstck,avg_phi_ge,c=colors[1],label=L"\Phi_{ge}",shape=:circle)
    plot!(p5,tstck,avg_phi_ke,c=colors[2],label=L"\Phi_{ke}",shape=:circle)
    plot!(p5,ylims=[0,avg_phi_ge[1]*1.2],ylabel=L"E/E[0]")

    plot!(p6,tstck,avg_w_me,c=colors[1],label=L"W_{ge}",shape=:circle)
    plot!(p6,tstck,avg_w_ee,c=colors[2],label=L"W_{ke}",shape=:circle)
    plot!(p6,ylims=[0,avg_w_me[1]*1.2],ylabel=L"E/E[0]")

    plot!(p7,tstck,avg_y_me,c=colors[1],label=L"Y_{ge}",shape=:circle)
    plot!(p7,tstck,avg_y_ee,c=colors[2],label=L"Y_{ke}",shape=:circle)
    plot!(p7,ylims=[0,avg_y_me[1]*1.2],ylabel=L"E/E[0]")

    plot(p5,p6,p7,layout=grid(3, 1),dpi=600)
    plot!(xlabel=L"m_Ht",xlims=[0,4])

    png(string(string(master_dir,"/",dir_name(para_arr[1])), "/ens-avg-thermalization-evo.png"))

    return
end

export energy_plots_cp
function energy_plots_cp(para_arr)
    
    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:tab10)
    p1=plot([],[],label="")
    p2=plot([],[],label="")
    p3=plot([],[],label="")
    p4=plot([],[],label="")
    p5=plot([],[],label="")
    p6=plot([],[],label="")
    p7=plot([],[],label="")

    for run_idx in range(1,size(para_arr,1),step=1)
        γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
        Nx = N
        
        run_dir = run_dirs[run_idx]
        total_energies= load_data(run_dir)

        box_vol = (Nx*dx)^3

        println(size(total_energies))#;exit()

        tot_energy_snp = [sum(total_energies[a,1:end-6]) for a in range(1,nsnaps+1,step=1)]./box_vol
        total_gauge_e = [sum(total_energies[a,4:7]) for a in range(1,nsnaps+1,step=1)]./box_vol
        total_phi_pot = total_energies[:,1]./box_vol
        total_phi_ke = total_energies[:,2]./box_vol
        total_phi_ge = total_energies[:,3]./box_vol
        total_w_ee = total_energies[:,4]./box_vol
        total_w_me = total_energies[:,5]./box_vol
        total_y_ee = total_energies[:,6]./box_vol
        total_y_me = total_energies[:,7]./box_vol
        total_b_eme = total_energies[:,8]./box_vol
        total_b2_eme = total_energies[:,9]./box_vol
        total_b3_eme = total_energies[:,11]./box_vol
        avgphi = total_energies[:,12]
        minphi = total_energies[:,10]
        tot_hel = total_energies[:,13]./box_vol

        # plot!(p1,range(0,nte,step=dsnaps).*dt*mH,(total_phi_pot.+total_phi_ke.+total_phi_ge)./tot_energy_snp[1],
        # label=run_idx)
        plot!(p1,range(0,nte,step=dsnaps).*dt*mH,tot_energy_snp./tot_energy_snp[1],
        label=run_idx)

        plot!(p2,range(0,nte,step=dsnaps).*dt*mH,minphi,xlabel=L"m_Ht",ls=:dashdot,label="")
        plot!(p2,range(0,nte,step=dsnaps).*dt*mH,avgphi,label=run_idx)

        t = range(0,nte,step=dsnaps).*dt*mH
        pe=(total_phi_pot.-lambda*vev^4).*(mH)^4
        ke=(total_phi_ke).*(mH)^4
        ge=(total_phi_ge).*(mH)^4
        me=(total_b_eme).*(mH)^4
        me2=(total_b2_eme).*(mH)^4
        total_energy = abs.(ke+ge+pe).-abs.(pe[1].+ke[1].+ge[1])
        plot!(p3,t,me./tot_energy_snp[1],label=run_idx)

        plot!(p4,range(0,nte,step=dsnaps).*dt*mH,minphi,xlabel=L"m_Ht",label=run_idx)
        # ptwin =twiny()
        # plot!(ptwin,range(0,nte,step=dsnaps).*dt/(Nx*dx),minphi,xlabel=L"t/T_{lc}")

        plot!(p5,range(0,nte,step=dsnaps).*dt*mH,tot_hel,label=run_idx)
        plot!(p5,xlims=[100,t[end]],ylims=[1e-6,1],yscale=:log)

    end

    # plot!(p1,framestyle=:box,ylims=[0.0,1.1])
    plot!(p1,framestyle=:box,ylims=[0.9,1.1])

    plot!(p1,ylabel=L"E(t)/E(0)")

    plot!(p2,framestyle=:box,ylims=[0.0,1.5])
    hline!(p2,[1,1],ls=:dot,color=:black,label="")

    plot(p1,p2,layout=grid(2, 1, heights=[0.6, 0.4]),dpi=600)
    png(string(master_dir,"/cp-plots/", "/cp-avg-higgs-evo.png"))

    plot!(p3,xlabel=L"m_H t",ylabel=L"{E_B}/{E(0)}")
    plot!(p3,ylims=[1e-3,1e-1])
    plot!(p3,grid=true,gridlinewidth=2.0,yscale=:log,dpi=600)
    plot!(p3,dpi=600)
    png(p3,string(master_dir,"/cp-plots","/cp-avg-mag-energy-evo.png"))

    plot!(p4,dpi=600)
    png(p4,string(master_dir,"/cp-plots","/cp-avg-min-phi.png"))

    plot!(p5,dpi=600,grid=true,xlabel=L"m_Ht",ylabel=L"H/V")
    png(p5,string(master_dir,"/cp-plots","/cp-hel-compare.png"))

    return
end

export energy_plots_gamma
function energy_plots_gamma(para_arr)
    
    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:tab10)
    p1=plot([],[],label="")
    p2=plot([],[],label="")
    p3=plot([],[],label="")
    p4=plot([],[],label="")
    p5=plot([],[],label="")
    p6=plot([],[],label="")
    p7=plot([],[],label="")

    for run_idx in range(1,size(para_arr,1),step=1)
        γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
        Nx = N
        
        run_dir = run_dirs[run_idx]
        total_energies= load_data(run_dir)

        box_vol = (Nx*dx)^3

        println(size(total_energies))#;exit()

        tot_energy_snp = [sum(total_energies[a,1:end-6]) for a in range(1,nsnaps+1,step=1)]./box_vol
        total_gauge_e = [sum(total_energies[a,4:7]) for a in range(1,nsnaps+1,step=1)]./box_vol
        total_phi_pot = total_energies[:,1]./box_vol
        total_phi_ke = total_energies[:,2]./box_vol
        total_phi_ge = total_energies[:,3]./box_vol
        total_w_ee = total_energies[:,4]./box_vol
        total_w_me = total_energies[:,5]./box_vol
        total_y_ee = total_energies[:,6]./box_vol
        total_y_me = total_energies[:,7]./box_vol
        total_b_eme = total_energies[:,8]./box_vol
        total_b2_eme = total_energies[:,9]./box_vol
        total_b3_eme = total_energies[:,11]./box_vol
        avgphi = total_energies[:,12]
        minphi = total_energies[:,10]
        tot_hel = total_energies[:,13]./box_vol

        # plot!(p1,range(0,nte,step=dsnaps).*dt*mH,(total_phi_pot.+total_phi_ke.+total_phi_ge)./tot_energy_snp[1],
        # label=run_idx)
        plot!(p1,range(0,nte,step=dsnaps).*dt*mH,tot_energy_snp./tot_energy_snp[1],
        label=γ)

        plot!(p2,range(0,nte,step=dsnaps).*dt*mH,minphi,xlabel=L"m_Ht",ls=:dashdot,label="")
        plot!(p2,range(0,nte,step=dsnaps).*dt*mH,avgphi,label=γ)

        t = range(0,nte,step=dsnaps).*dt*mH
        pe=(total_phi_pot.-lambda*vev^4).*(mH)^4
        ke=(total_phi_ke).*(mH)^4
        ge=(total_phi_ge).*(mH)^4
        me=(total_b_eme).*(mH)^4
        me2=(total_b2_eme).*(mH)^4
        total_energy = abs.(ke+ge+pe).-abs.(pe[1].+ke[1].+ge[1])
        plot!(p3,t,me./tot_energy_snp[1],label=γ)

        plot!(p4,range(0,nte,step=dsnaps).*dt*mH,minphi,xlabel=L"m_Ht",label=γ)
        # ptwin =twiny()
        # plot!(ptwin,range(0,nte,step=dsnaps).*dt/(Nx*dx),minphi,xlabel=L"t/T_{lc}")

        plot!(p5,range(0,nte,step=dsnaps).*dt*mH,tot_hel,label=γ)
        plot!(p5,xlims=[100,t[end]],ylims=[1e-6,1],yscale=:log)

    end

    # plot!(p1,framestyle=:box,ylims=[0.0,1.1])
    plot!(p1,framestyle=:box,ylims=[0.5,1.02])

    plot!(p1,ylabel=L"E(t)/E(0)")

    plot!(p2,framestyle=:box,ylims=[0.0,1.5])
    hline!(p2,[1,1],ls=:dot,color=:black,label="")

    plot(p1,p2,layout=grid(2, 1, heights=[0.6, 0.4]),dpi=600)
    png(string(master_dir,"/gamma-plots/", "/gamma-avg-higgs-evo.png"))

    plot!(p3,xlabel=L"m_H t",ylabel=L"{E_B}/{E(0)}")
    plot!(p3,ylims=[1e-3,1e-1])
    plot!(p3,grid=true,gridlinewidth=2.0,yscale=:log,dpi=600)
    plot!(p3,dpi=600)
    png(p3,string(master_dir,"/gamma-plots","/gamma-avg-mag-energy-evo.png"))

    plot!(p4,dpi=600)
    png(p4,string(master_dir,"/gamma-plots","/gamma-avg-min-phi.png"))

    plot!(p5,dpi=600,grid=true,xlabel=L"m_Ht",ylabel=L"H/V")
    png(p5,string(master_dir,"/gamma-plots","/gamma-hel-compare.png"))

    return
end


end
