## SLURM CLI Tools

```bash
# admin gui
sview

# node availables
sinfo

# see jobs in queue
squeue

# see resource allocation by other jobs
squeue -o "%c %u %e %m %T %b"

# execute a job from a script
sbatch my_script.sh

# get infos about a job
scontrol show job JOB_ID
```

## Example

First, copy paste the `/storage/documentation/slurm_job_example/` folder to your `$HOME`: `cp -R /storage/documentation/slurm_job_example/ $HOME/slurm_job_example/`.

**IMPORTANT:** the example SLURM job assume you have **a working mamba installation** and also created a conda env called `my_conda_env`. Refer to `./USER_INSTALL_MAMBA.md`.

Check the `$HOME/slurm_job_example/job.sh` file for details about how to configure your SLURM job.

Then you can execute your job and monitor it with:

```bash
cd $HOME/slurm_job_example/

sbatch job.sh

# check the SLURM queue
squeue

# check the logs in real-time
tail -f job.out
```

## Interesting commands

```
# Watch the squeue in real-time and refresh every second
watch -n 1 squeue

# Check only your jobs
squeue -u $USER

# Cancel a job
scancel <jobid>

```

## Configuring your Bash file

You will notice a few lines of commented code at the beginning of your `.sh` file that look like configurations.
This is because `sbatch` is able to recognizes lines that start with `#SBATCH --OPTION`. If you want a line to be ignored, simply add a second `#` symbol.

Here, you have a list of common options.

- `job-name`: Name of your SLURM job
- `array`: Run the same job many times with the same option. 
  - The format `0-5` is equivalent to a file 6 times with values 0, 1, 2, ..., 5
  - You can then use this for hyper-parameter search. For example, defining 10 sets of h-params, then using the variable `$SLURM_ARRAY_TASK_ID` to select the right set of h-params.
  - You can specify `0-100%25` for 100 jobs, where only 25 run simultaneously
- `output`: Name of the output file where logs are printed. You can add the following special characters
  - `%x`: Take the `job-name` option and add it in the name of the output file
  - `%j`: Take the `job-id` and add it in the name of the output file
  - `%a`: Take the `array-id` and add it in the name of the output file
- `error`: Same as output, but for saving the error logs. It can have the same value as `output`.
- `time`: Time limit to run the job. In the format `HH:MM:SS`
- `mem`: Memory limit to run a job, in MB.
- `gres`: Options regarding the GPUs
  - `--gres=gpu:X` Use `X` number of GPUs. `0` for no GPU and `1` for a single GPU.
  - `--gres=mps:X` Use `X`% fraction of GPUs, in percentage.

Check out [this cheat sheet](http://www.physik.uni-leipzig.de/wiki/files/slurm_summary.pdf) for more commands and information.

When selecting the resources that you need, please be mindful that others might be using the same machine to run other jobs. So don't be too greedy on resource counts. Don't select more RAM, CPU, or GPU than needed.

## Splitting a GPU for multiple jobs
Using the option `--gres=mps:X` in your bash file allows you to split your GPU into `1/X` parts, where `X` is given in percentage. For eample, `--gres=mps:20` means that you want to use only 20% of your GPU for a job. This allows you to run other jobs on the same GPU. 

Multiple jobs can be launched on the same GPU either from the `array` option, or simply by running `sbatch` many times. Multiple users can use the `mps` option simultaneously, but it will be on the same GPU.

However, there are limitations that are directly due to NVidia [CUDA_Multi_Process_Service_Overview](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf).
- **On a single machine, the `mps` can only be used on a Single GPU. Given 3 GPUs, Bob cannot run 6 jobs, each on 50% of a GPU.**
- **Once Bob submits a job using `mps`, he cannot submit any other job requiring GPUs, with or without MPS.**
- There might be other limitations, please report them here.

One could bypass these limitations by simply launching many python files from the job. This should work (I think), but without strict allocation of resources, the scripts might compete for the same resources and cause an out-of-memory error. Please, report here if this option doesn't work.

## Interactive sessions for debugging and jupyter notebooks?
salloc runs interactive jobs on slurm. You can allocate the recourses you need without running sbatch job and keep the resource as if it is on your laptop. This is great for prototyping code. 

```
#Andy's favourite salloc command 
salloc --cpus-per-task=4  --gres=gpu:1 --mem=32G

#number of task in your script, usually=1
-n, -ntasks=<number>

#The number of cores for each task
-c, –cpus-per-task=<ncpus>

#Time requested for your job
-t, –time=<time>

#Memory requested for all your tasks
–mem=<size[units]>

#Select generic resources such as GPUs for your job: --gres=gpu:GPU_MODEL
–gres=<list>

#to leave the interactive job use the `exit` command or just close the terminal 
exit

#you can also see the interactive job in squeue 
squeue -u $USER
```
## Single line jupyter session using `srun`.

```bash
# Configure the SLURM options
SLURM_OPTIONS="--pty --time=2:00:00 --mem=64G --cpus-per-task=16 --ntasks=1"

# Start a jupyter session in interactive mode
srun $SLURM_OPTIONS -- micromamba run -n CONDA_ENV_NAME jupyter lab --no-browser --IdentityProvider.token="" --ip=0.0.0.0 --port=8888
```

Then, you can access your Jupyter session using a VSCode port proxy or a more regular SSH proxy to rabelais on the correct Jupyter port. 

## Hyper-parameter search with Arrays and Click

Using SLURM-arrays and Click, it's easy to do a hyper-parameter search on a single GPU, with an example below. With SLURM-arrays, you can run many jobs with exactly the same parameters, except for an array-id that is provided. This array-id is equivalent to the `ii` parameter in `for ii in range(X)`. By using the `click.option`, you can pass this `array_id` parameter in your python script and select the right hyper-parameters.

Example of a `job_hparams.sh` file, running 8 jobs with ``--array=0-7``, each using 20% of a GPU.

```
#!/usr/bin/env bash
#SBATCH --job-name=hparam_search
#SBATCH --output=%x-%j-%a.out
#SBATCH --error=%x-%j-%a.out
#SBATCH --open-mode=append
#SBATCH --time=10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000
#SBATCH --array=0-7
#SBATCH --gres=mps:20

set -e
source activate my_conda_env
python run_deep_nn.py --array_id=$SLURM_ARRAY_TASK_ID
sleep 20s
```

This will call the python file `run_deep_nn.py` with the parameter `array_id`. An example of such file is defined below. You will see that the `DeepNN` class is created with a different set of hyper-parameters for each run.

```
import click
import torch

class DeepNN(torch.nn.Module):
    def __init__(self, hparams):
        self.hparams = hparams
    def forward(self):
        print(f'hparams: {self.hparams}')

HYPER_PARAMETERS = [
    {"hidden_dim": 32, "depth": 4, "seed": 1},
    {"hidden_dim": 64, "depth": 4, "seed": 1},
    {"hidden_dim": 32, "depth": 2, "seed": 1},
    {"hidden_dim": 64, "depth": 2, "seed": 1},
    {"hidden_dim": 32, "depth": 4, "seed": 2},
    {"hidden_dim": 64, "depth": 4, "seed": 2},
    {"hidden_dim": 32, "depth": 2, "seed": 2},
    {"hidden_dim": 64, "depth": 2, "seed": 2},
]

@click.command()
@click.option('--array_id', default=None)
def my_neural_network(array_id):
    array_id = int(array_id)
    print(f"array_id {array_id}")
    deep_nn = DeepNN(HYPER_PARAMETERS[array_id])
    deep_nn.forward()

if __name__ == '__main__':
    my_neural_network()
```

