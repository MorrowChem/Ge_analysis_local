#!/bin/bash
#SBATCH --job-name=gap_fit
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --account=chem-amais
module load gcc/7.3.0
at_file=../train_216_125_64_v.xyz gap={distance_Nb order=2 cutoff=5.0 covariance_type=ARD_SE theta_uniform=2.0 n_sparse=15 delta=2.0 sparse_method=uniform compact_clusters=T : soap l_max=6 n_max=12 atom_sigma=0.5 zeta=4 cutoff=5.0 cutoff_transition_width=1.0 central_weight=1.0 n_sparse=5000 delta=0.155 f0=0.0 covariance_type=dot_product sparse_method=CUR_POINTS} energy_parameter_name=dft_energy force_parameter_name=dft_forces virial_parameter_name=NOT_USED sparse_jitter=1.0e-8 default_sigma={0.002 0.2 0.0 0.0} do_copy_at_file=F sparse_separate_file=T gap_file=as5_ds002_vT 