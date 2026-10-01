# Sourced by the CD1-L launchers: GPU stack on ACCRE (A6000/A4000, cc 8.6).
# STACK = checkout holding .venv and repos/; REPO = this lisa-analysis-tools tree.
STACK=${STACK:-/home/bustam1/lisastack_pr81}
REPO=${REPO:-$STACK/repos/LISAanalysistools}
CUDA_ROOT="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.6.2"
module --force purge
module load StdEnvACCRE/2023; module load python/3.13.2; module load mpi4py gsl
export CUDA_HOME=${CUDA_ROOT}; export PATH=${CUDA_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/openmpi/4.1.5/lib:${LD_LIBRARY_PATH:-}
export FEW_BACKEND=cuda12x
# activate (not .venv/bin/python): it puts site-packages/nvidia/*/lib on
# LD_LIBRARY_PATH, without which jax silently falls back to CPU.
source "$STACK/.venv/bin/activate"
python -c "import jax; assert jax.default_backend()=='gpu', 'jax on CPU'" || exit 1
cd "$REPO"
export CD1L_ROOT=${CD1L_ROOT:-/data/nbody/majoburo/cd1l_pe}
