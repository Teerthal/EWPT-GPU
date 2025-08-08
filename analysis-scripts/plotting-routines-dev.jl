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

include("data-routines.jl")
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
    k_max = k_phys_binned[argmax(Bk_phys_binned[findall(@. !isnan(Bk_phys_binned))])]
    
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
    idxs=[(1)]


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
    plot!(p1,[],[],label="non-binned",seriestype=:scatter,c=:black)
    plot!(p1,[],[],label="binned",c=:black)
    plot!(p1,[],[],label=L"$p_{mean}$",c=:black,ls=:dash)
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
        idxs=[(10+1),no_fft_snaps-max_spec_t_idx]

        time_stamps = time_stamps
        dt = dt 

        time_plot_idxs = idxs
        # idxs=[(no_fft_snaps+1)÷2,no_fft_snaps-max_spec_t_idx]
        iter=1
        # for idx in range(2,size(restacked_fft,1),step=1)
        spec_stack_t = []
        for idx in idxs

            k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(run_dirs[run_idx],idx,run_idx)
            k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        
            k_phys,Hk_phys,k_phys_binned,Hk_phys_binned,Hk_phys_binned_err=phys_conv(H_fft_re,H_fft_re_binned,dx,Nx)

            ##E_M spectrum plot##
            plot!(p1,k_phys_binned,Bk_phys_binned,msc=colors[iter],c=colors[iter],
            label="",lw=0.5,alpha=0.5)
            push!(spec_stack_t,Bk_phys_binned)
            kplot = k_phys_binned
            iter=iter+1
        end
        push!(spec_stack,spec_stack_t)
        
    end

    ##ensemble averaged spectra##
    iter = 1
    for idx in time_plot_idxs

        # println(size([spec_stack[i][iter] for i in range(1,no_runs,step=1)]))
        # plot!(p1,kplot,mean([spec_stack[i][iter] for i in range(1,no_runs,step=1)]),yerr=std([spec_stack[i][iter] for i in range(1,no_runs,step=1)])./sqrt(no_runs),msc=colors[iter],c=colors[iter],
        # label="",lw=1.5)

        plot!(p1,kplot,mean([spec_stack[i][iter] for i in range(1,size(spec_stack,1),step=1)]),msc=colors[iter],c=colors[iter],
        label=string(round(time_stamps[idx]*dt*mH,digits=0)),lw=1.5)

        iter = iter+1
    end
    plot!(p1,xlims=[0,k_xlim_upper],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    plot!(p1,dpi=600)
    plot!(p1,xlabel=L"p/\eta",ylabel=L"4\pi p^2|B(p)|^2/V")
    plot!(p1,minorgrid=true,yscale=:linear,framestyle=:box)
    png(p1,string(string(master_dir,"/",dir_name(para_arr[1])),"/post-spec-t-combined-linear-ensemble.png"))

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


export fft_plots_log_therm
function fft_plots_log_therm(para_arr;no_runs=25)
    
    gr()
    ENV["GKSwstype"]="nul"
    colors=palette(:tab10)
    p1=plot([],[],label="")
    p2=plot([],[],label="")
    plot!(p1,[],[],label="non-binned",seriestype=:scatter,c=:black)
    plot!(p1,[],[],label="binned",c=:black)
    plot!(p1,[],[],label=L"$p_{mean}$",c=:black,ls=:dash)
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
        idxs=[(10+1),(20+1),no_fft_snaps-max_spec_t_idx]

        time_plot_idxs = idxs 
        # idxs=[(no_fft_snaps+1)÷2,no_fft_snaps-max_spec_t_idx]
        iter=1
        # for idx in range(2,size(restacked_fft,1),step=1)
        spec_stack_t = []
        for idx in idxs

            k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(run_dirs[run_idx],idx,run_idx)
            k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        
            k_phys,Hk_phys,k_phys_binned,Hk_phys_binned,Hk_phys_binned_err=phys_conv(H_fft_re,H_fft_re_binned,dx,Nx)

            ##E_M spectrum plot##
            plot!(p1,k_phys_binned[findall(k_phys_binned.!=0)],Bk_phys_binned[findall(k_phys_binned.!=0)],msc=colors[iter],c=colors[iter],
            label="",lw=0.5,alpha=0.5)
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
        fit = curve_fit(m, log.(kplot[1:low_k_cut_idx]), log.(Bk_phys_binned_mean[1:low_k_cut_idx]), p0)
        plot!(p1,kplot[1:low_k_cut_idx],exp.(m(log.(kplot[1:low_k_cut_idx]),fit.param)),
        ls=:dashdot,c=colors[iter],label=string(L"$\alpha_1$:",round(fit.param[1],digits=1)))

        ##High k fit
        p0 = [-4.0, 1.0]

        high_k_cut_idx = findall(Bk_phys_binned_mean.==maximum(Bk_phys_binned_mean))[1]
        high_k_cut_upp_idx =(findall(Bk_phys_binned_mean.>maximum(Bk_phys_binned_mean)/3))[end]

        fit = curve_fit(m, log.(kplot[high_k_cut_idx:high_k_cut_upp_idx]), log.(Bk_phys_binned_mean[high_k_cut_idx:high_k_cut_upp_idx]), p0)
        plot!(p1,kplot[high_k_cut_idx:high_k_cut_upp_idx],exp.(m(log.(kplot[high_k_cut_idx:high_k_cut_upp_idx]),fit.param)),
        ls=:dashdot,c=colors[iter*3],label=string(L"$\alpha_2$:",round(fit.param[1],digits=1)))
        
        vline!(p1,[kplot[high_k_cut_upp_idx]],c=colors[iter*3],ls=:dashdot,label="")

        ###Very high k fit###
        veryhigh_k_cut_upp_idx = (findall(Bk_phys_binned_mean.>maximum(Bk_phys_binned_mean/1e4)))[end]
        fit = curve_fit(m, log.(kplot[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]), log.(Bk_phys_binned_mean[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]), p0)
        plot!(p1,kplot[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx],exp.(m(log.(kplot[high_k_cut_upp_idx:veryhigh_k_cut_upp_idx]),fit.param)),
        ls=:dashdot,c=colors[iter*4],label=string(L"$\alpha_3$:",round(fit.param[1],digits=1)))
        
        vline!(p1,[kplot[high_k_cut_upp_idx]],c=colors[iter*4],ls=:dashdot,label="")

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
    
    plot!(p1,xlims=[min_k,k_xlim_upper],ylims=[max_B_k/20,max_B_k],bottom_margin=0*Plots.mm,top_margin=0*Plots.mm)
    plot!(p1,xscale=:log,yscale=:log)
    plot!(p1,dpi=600)
    plot!(p1,xlabel=L"p/\eta",ylabel=L"4\pi p^2|B(p)|^2/V")
    plot!(p1,minorgrid=true,framestyle=:box)
    png(p1,string(string(master_dir,"/",dir_name(para_arr[1])),"/post-spec-t-combined-log-ensemble.png"))


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

export energy_plots
function energy_plots(para_arr;run_idx=1)
    γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
    run_dir = run_dirs[run_idx]
    total_energies= load_data(run_dir)
    Nx = N
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

    gr()
    ENV["GKSwstype"]="nul"
    plot(range(0,nte,step=dsnaps).*dt*mH,(total_phi_pot.+total_phi_ke.+total_phi_ge)./tot_energy_snp[1],
    label="Higgs")
    plot!(range(0,nte,step=dsnaps).*dt*mH,(total_w_ee.+total_w_me)./tot_energy_snp[1],
    label="SU(2)")
    plot!(range(0,nte,step=dsnaps).*dt*mH,(total_y_ee.+total_y_me)./tot_energy_snp[1],
    label="U(1)")
    plot!(range(0,nte,step=dsnaps).*dt*mH,total_b_eme./tot_energy_snp[1],
    label=L"B^{(YY)}",ls=:dot)
    plot!(range(0,nte,step=dsnaps).*dt*mH,total_b2_eme./tot_energy_snp[1],
    label=L"B^{(TV)}")
    plot!(range(0,nte,step=dsnaps).*dt*mH,tot_energy_snp./tot_energy_snp[1],
    label="Total")
    plot!(ylabel=L"\epsilon(t)/\epsilon(0)")
    ptwin =twinx()
    plot!(ptwin,range(0,nte,step=dsnaps).*dt*mH,tot_energy_snp./tot_energy_snp[1],label="")
    p1=plot!(minorgrid=true,ylims=[-0.1,1.1],framestyle=:box)

    p2=plot(range(0,nte,step=dsnaps).*dt*mH,minphi,xlabel=L"m_Ht",label="",ls=:dashdot)
    # plot!(p2,range(0,nte,step=dsnaps).*dt*mH,sqrt.(-sqrt.(total_phi_pot./lambda).+1.0),xlabel=L"m_Ht",ls=:dot)
    ptwin =twinx()
    plot!(ptwin,range(0,nte,step=dsnaps).*dt*mH,minphi,label="",ls=:dashdot)
    plot!(p2,range(0,nte,step=dsnaps).*dt*mH,avgphi,label="")
    plot!(p2,framestyle=:box,ylims=[0.0,1.5])
    hline!([1,1],ls=:dot,color=:black,label="")
    plot(p1,p2,layout=grid(2, 1, heights=[0.8, 0.2]),dpi=600)
    png(string(run_dir, "/post-total-energies-components-2.png"))

    # exit()

    t = range(0,nte,step=dsnaps).*dt*mH
    pe=(total_phi_pot.-lambda*vev^4).*(mH)^4
    ke=(total_phi_ke).*(mH)^4
    ge=(total_phi_ge).*(mH)^4
    me=(total_b_eme).*(mH)^4
    me2=(total_b2_eme).*(mH)^4
    total_energy = abs.(ke+ge+pe).-abs.(pe[1].+ke[1].+ge[1])

    gr()
    ENV["GKSwstype"]="nul"
    plot(t,[pe ke ge me2 total_energy],label=["PE" "KE" "GE" L"ME\times10^6" "Total"])
    # plot!(twinx(),[pe ke ge total_energy],legend=false)
    plot!(xlabel=L"m_H t",ylabel=L"{<E>}/{m_H^4}")
    plot!(grid=true,gridlinewidth=2.0,xlims=[0,200],ylims=[-0.05,0.05],dpi=600)
    png(string(run_dir,"/energies-evo.png"))

    gr()
    ENV["GKSwstype"]="nul"
    plot(t,[me,me2],label=["ME" "ME2"])
    # plot!(twinx(),[pe ke ge total_energy],legend=false)
    plot!(xlabel=L"m_H t",ylabel=L"{<E>}/{m_H^4}")
    plot!(grid=true,gridlinewidth=2.0,yscale=:log,dpi=600)
    png(string(run_dir,"/mag-energy-evo.png"))


    gr()
    ENV["GKSwstype"]="nul"
    plot(range(0,nte,step=dsnaps).*dt*mH,[total_phi_pot total_phi_ke total_phi_ge total_gauge_e total_b_eme tot_energy_snp],
            label=["PE" "KE" "GE" "Gauge" "B" "Total"],xlims=(0,nte.*dt*mH),dpi=600)
    png(string(run_dir,"/post-energies-linear.png"))

    gr()
    ENV["GKSwstype"]="nul"
    plot(range(0,nte,step=dsnaps).*dt*mH,minphi,xlabel=L"m_Ht")
    ptwin =twiny()
    plot!(ptwin,range(0,nte,step=dsnaps).*dt/(Nx*dx),minphi,xlabel=L"t/T_{lc}")
    plot!(dpi=600)
    png(string(run_dir,"/post-min-phi.png"))

    # gr()
    # ENV["GKSwstype"]="nul"
    # plot(range(0,nte,step=dsnaps).*dt/(Nx*dx),sqrt.(-sqrt.(total_phi_pot./lambda).+1.0),xlabel=L"m_Ht")
    # ptwin =twiny()
    # plot!(ptwin,range(0,nte,step=dsnaps).*dt/(Nx*dx),sqrt.(-sqrt.(total_phi_pot./lambda).+1.0),xlabel=L"t/T_{lc}")
    # plot!(dpi=600)
    # png(string(run_dir,"/post-avg-phi.png"))

    gr()
    ENV["GKSwstype"]="nul"
    plot(range(0,nte,step=dsnaps).*dt*mH,total_phi_pot.+total_phi_ke.+total_phi_ge,
    label="Higgs")
    plot!(range(0,nte,step=dsnaps).*dt*mH,total_w_ee.+total_w_me,
    label="SU(2)")
    plot!(range(0,nte,step=dsnaps).*dt*mH,total_y_ee.+total_y_me,
    label="U(1)")
    plot!(range(0,nte,step=dsnaps).*dt*mH,total_b_eme,
    label=L"B^{(YY)}",ls=:dot)
    plot!(range(0,nte,step=dsnaps).*dt*mH,total_b2_eme,
    label=L"B^{(TV)}")
    plot!(range(0,nte,step=dsnaps).*dt*mH,tot_energy_snp,
    label="Total")
    p1=plot!(dpi=600,minorgrid=true,ylabel=L"\epsilon/\eta^4",ylim=[0,maximum(tot_energy_snp)],framestyle=:box)
    p2=plot(range(0,nte,step=dsnaps).*dt*mH,minphi,dpi=600,xlabel=L"m_Ht",ls=:dashdot,label=L"min(\|\Phi\|)")

    ptwin =twinx()
    plot!(ptwin,range(0,nte,step=dsnaps).*dt*mH,minphi,ls=:dashdot,label="")
    plot!(p2,range(0,nte,step=dsnaps).*dt*mH,avgphi,label=L"\langle(\|\Phi\|)\rangle")
    plot!(p2,framestyle=:box,ylims=[0.0,1.5])
    # ptwin =twiny()
    # plot!(ptwin,range(0,nte,step=dsnaps).*dt/(Nx*dx),minphi,xlabel=L"t/T_{lc}",ls=:dashdot)
    hline!([1,1],ls=:dot,color=:black,label="")
    plot(p1,p2,layout=grid(2, 1, heights=[0.8, 0.2]),dpi=600)



    png(string(run_dir, "/post-total-energies-components.png"))


    gr()
    ENV["GKSwstype"]="nul"
    plot(range(0,nte,step=dsnaps).*dt*mH,total_b_eme,
    label=L"B^{(YY)}")
    plot!(range(0,nte,step=dsnaps).*dt*mH,total_b2_eme,
    label=L"B^{(TV)}")
    plot!(range(0,nte,step=dsnaps).*dt*mH,total_b3_eme,
    label=L"B^{(Mou)}")
    p1=plot!(dpi=600,minorgrid=true,ylabel=L"\epsilon/\eta^4",ylim=[0,maximum(total_b_eme)*2])
    p2=plot(range(0,nte,step=dsnaps).*dt*mH,minphi,dpi=600,xlabel=L"m_Ht",ls=:dashdot,label=L"min(\|\Phi\|)")
    plot!(p2,range(0,nte,step=dsnaps).*dt*mH,avgphi,label=L"\langle(\|\Phi\|)\rangle")

    ptwin =twinx()
    plot!(ptwin,range(0,nte,step=dsnaps).*dt*mH,minphi,ls=:dashdot,label="")
    plot!(p2,framestyle=:box,ylims=[0.0,1.5])
    hline!([1,1],ls=:dot,color=:black,label="")
    plot(p1,p2,layout=grid(2, 1, heights=[0.8, 0.2]),dpi=600)

    png(string(run_dir, "/post-mag-energies.png"))
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

    #select good runs ffor 320 runs#
    sub_runs_plots = filter(x -> x ∉ [8,22,23], range(1,no_runs,step=1))
    #select good runs for N=256#
    sub_runs_plots = filter(x -> x ∉ [60], range(1,no_runs,step=1))

    for run_idx in sub_runs_plots
        γ,no_bubbles,nte,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
        Nx = N
        time_stamps = time_stamps
        dt = dt 
        run_stack = []
        t_arr = []
        time_start = 1

        plot_time_idxs = range(time_start,no_fft_snaps-max_spec_t_idx,step=1)

        for idx in plot_time_idxs
    
            k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,H_k_points,H_fft_re,stacked_fft_H,H_fft_re_binned=load_processed_fft_data(run_dirs[run_idx],idx,run_idx)
            k_phys,Bk_phys,k_phys_binned,Bk_phys_binned,Bk_phys_binned_err=phys_conv(B_fft_re,B_fft_re_binned,dx,Nx)        
            k_phys,Hk_phys,k_phys_binned,Hk_phys_binned,Hk_phys_binned_err=phys_conv(H_fft_re,H_fft_re_binned,dx,Nx)
    
            # peaks_arr = find_mean_peaks(restacked_fft)
            t,k_mean_nonbinned,k_mean_binned,k_max,xi_mean_nonbinned,xi_mean_binned = find_mean_peaks(k_points,B_k_points,B_fft_re,stacked_fft,B_fft_re_binned,idx,run_idx)
            push!(run_stack,xi_mean_binned)
            push!(t_arr,time_stamps[idx]*dt*mH)
            
        end
        
        if size(run_stack,1)==size(plot_time_idxs,1)
            println(string(run_idx,", ", size(run_stack,1), " ", size(plot_time_idxs,1)," ", size(run_stack)))
            push!(peaks_arr,run_stack)
        end
        
        t=t_arr
    end
    println(size(peaks_arr))
    len_xi = size(peaks_arr[1],1)
    xi_mean = [mean([peaks_arr[i][iter] for i in range(1,size(peaks_arr,1),step=1)]) for iter in range(1,len_xi,step=1)]
    # println(xi_mean);exit()
    plot!(p1,t,xi_mean)
  
    plot!(p1,xscale=:linear,yscale=:linear)
    plot!(p1,dpi=600)
    plot!(p1,minorgrid=true,framestyle=:box)
    png(p1,string(string(master_dir,"/",dir_name(para_arr[1])),"/post-evo-plots-ensemble.png"))

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

    time_stamps = []
    data_arr = []
    t =[]
    dt = 0.0

    # #select good runs for 320 runs#
    # sub_runs_plots = filter(x -> x ∉ [8,22,23], range(1,no_runs,step=1))
    # #select good runs for N=256#
    # sub_runs_plots = filter(x -> x ∉ [60], range(1,no_runs,step=1))
    ## plot all without spnge,dx,dt,N,gp2,nsnaps,T,dsnaps,dsnaps_fft,no_fft_snaps,time_stamps = paras(run_dirs[run_idx],para_arr[run_idx])
        run_dir = run_dirs[run_idx]
        total_energies= load_data(run_dir)
        Nx = N
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

        plot!(p1,range(0,nte,step=dsnaps).*dt*mH,(total_phi_pot.+total_phi_ke.+total_phi_ge)./tot_energy_snp[1],
        label=run_idx)

        plot!(p2,range(0,nte,step=dsnaps).*dt*mH,minphi,xlabel=L"m_Ht",ls=:dashdot)
        plot!(p2,range(0,nte,step=dsnaps).*dt*mH,avgphi,label="",label=run_idx)

        plot!(p3,t,me,label=run_idx)

        plot!(p4,range(0,nte,step=dsnaps).*dt*mH,minphi,xlabel=L"m_Ht",label=run_idx)
        # ptwin =twiny()
        # plot!(ptwin,range(0,nte,step=dsnaps).*dt/(Nx*dx),minphi,xlabel=L"t/T_{lc}")
    end

    plot!(p1,ylabel=L"\rho_{H}(t)/\rho_{H}(0)")

    plot!(p2,framestyle=:box,ylims=[0.0,1.5])
    hline!(p2,[1,1],ls=:dot,color=:black,label="")

    plot(p1,p2,layout=grid(2, 1, heights=[0.6, 0.4]),dpi=600)
    png(string(string(master_dir,"/",dir_name(para_arr[1])), "/ens-avg-higgs-evo.png"))

    plot!(p3,xlabel=L"m_H t",ylabel=L"{<E>}/{m_H^4}")
    plot!(p3,grid=true,gridlinewidth=2.0,yscale=:log,dpi=600)
    plot!(p3,dpi=600)
    png(p3,string(string(master_dir,"/",dir_name(para_arr[1])),"/ens-avg-mag-energy-evo.png"))


    plot!(p4,dpi=600)
    png(p4,string(string(master_dir,"/",dir_name(para_arr[1])),"/ens-avg-min-phi.png"))

    return
end

end
