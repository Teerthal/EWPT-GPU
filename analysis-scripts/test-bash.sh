N=1

for i in {1..51}; do        ##slurm-array-task-id
    (
        # .. do your stuff here
        echo "starting task $i.."
        # sleep $(( (RANDOM % 3) + 1))
        julia --project --check-bounds=no -O3 general-reader.jl 1 ${i} 7
    ) &

    # allow to execute up to $N jobs in parallel
    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        # now there are $N jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
    fi

done