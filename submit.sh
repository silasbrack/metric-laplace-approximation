#!/bin/sh
### General options

### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J metric-learning

### -- ask for number of cores (default: 1) --
#BSUB -n 4
##BSUB -R "span[hosts=1]"

### -- Select the resources: 1 gpu in exclusive process mode -- 
##BSUB -cpu "num=8"
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# request 5GB of system-memory
##BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=16GB]"
##BSUB -R "select[model == XeonGold6226R]"

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u moe.simon@gmail.com
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --

#BSUB -o logs/runid-%J.out
#BSUB -e logs/runid-%J.err
# -- end of LSF options --

# Load the cuda module
module load python3/3.9.6
module load cuda/11.3

# Go to directory
cd /zhome/e2/5/127625/PycharmProjects/metric-laplace-approximation

# Load venv
source venv/bin/activate

# Run test
python3 -m src.models.train_laplace --epochs=20 --hessian="full"

