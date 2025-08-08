#!/bin/bash

# gamma=0.0
# no_bubbles=10
method="ker-rk4"
gamma="$1"
no_bubbles="$2"
nte="$3"
dx="$4"
_dt="$5"
latx="$6"
gp2="$7"
nsnaps="$8"
T="$9"

echo "No bubbles: " $no_bubbles

master_dir=/home/tpatel28/topo_mag/EW_sim
echo $master_dir

echo "Gamma: " $gamma

code_dir=kernels-rk4
cd ${master_dir}/runs/many-bubbles-test/scaling-ker4
run_dir=gamma-${gamma}-bubbles-${no_bubbles}-nte-${nte}-dx-${dx}-_dt-${_dt}-latx-${latx}-gp2-${gp2}-nsnaps-${nsnaps}-T-${T}
mkdir $run_dir
cp ${master_dir}/${code_dir}/*.jl ${master_dir}/runs/many-bubbles-test/scaling-ker4/${run_dir}/
cp ${master_dir}/${code_dir}/*.sh ${master_dir}/runs/many-bubbles-test/scaling-ker4/${run_dir}/
cd ${master_dir}/runs/many-bubbles-test/scaling-ker4/${run_dir}
sbatch run.sh ${gamma} ${no_bubbles} ${nte} ${dx} ${_dt} ${latx} ${gp2} ${nsnaps} ${T}
