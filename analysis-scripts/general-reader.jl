using Plots     ##Needs to be loaded before HDF5??##
using WriteVTK
using HDF5
using Statistics
using StatsBase
using FFTW
using LaTeXStrings
using LsqFit
using Glob
using Distributed

include("data-routines.jl")
using .data_managers

include("plotting-routines.jl")
using .plotters

# println(run_dirs)


if (run_type=="thermal_ens_average")
    # fft_plots_linear_therm(para_arr,no_runs=25)
    # fft_plots_log_therm(para_arr,no_runs=25)
    # fft_plots_log_therm_early(para_arr,no_runs=25)
    # energy_plots_therm_early(para_arr,no_runs=25)
    peak_evo__therm(para_arr,no_runs=25)
    # energy_plots_therm(para_arr,no_runs=25)
else
    # fft_plots_linear(para_arr,run_idx=1)
    # exit()
    # fft_plots_log(para_arr)
    
    prep_therm(para_arr,idx=parse(Int,ARGS[2]))
    # prep_therm_all(para_arr)

    # peak_evo_plot(para_arr)
    
    # energy_plots(para_arr)

    # energy_plots_cp(para_arr)
end

exit()
