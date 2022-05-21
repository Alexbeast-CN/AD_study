import os
import multiprocessing
import argparse
import json
from slurm_template import slurm_template
from datetime import datetime
import subprocess


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--hyperparams", default="carracing_hyperparams.json")
    parser.add_argument("--slurm_settings", default="slurm_settings.json",
                        help="file of SLURM specific options (e.g. number of GPUS)")
    parser.add_argument("--cluster", default=False, action="store_true")
    parser.add_argument("--start", type = int, default=0,
                        help="the start index of experiments")
    parser.add_argument("--end", type = int, default=None,
                        help="the end index of experiments")
    
    args = parser.parse_args()

    # load the hyperparameters
    with open(args.hyperparams, "r") as f:
        hyperparams = json.load(f)
      
    exp_flags = []
    for exp in hyperparams:
        exp_flags.append(" ".join([f"--{key} {val}" if val is not "store_true" else f"--{key}" for key, val in exp.items()]))
    exp_flags = exp_flags [args.start:args.end]
    print(exp_flags)

    # load SLURM specific settings
    with open(args.slurm_settings, "r") as f:
        slurm_default = json.load(f)
    print ( slurm_default )
    
    training_in_cluster = args.cluster

    # make a folder for the current trial
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    main_dir = "experiments_" + current_time
    os.makedirs ( main_dir, exist_ok=True )
    
    for idx, exp_flag in enumerate(exp_flags):
        
        # make a directory for each experiment
        exp_name = "experiment_{0}".format( idx )
        exp_dir = os.path.join ( main_dir, exp_name )
        os.makedirs ( exp_dir, exist_ok=True )
        
        # for stdout of slurm jobs.
        job_errfile = os.path.join ( exp_dir, "job_{0}.err".format ( idx ) )
        job_outfile = os.path.join ( exp_dir, "job_{0}.out".format ( idx ) )

        additional_args = " --agent_name " + "agent_{0}".format( idx ) + " --outdir " + exp_dir 
        command = "singularity exec --nv ~/sdc_new_gym21.simg python train_racing.py " + exp_flag + additional_args
        #print ( command )

        if not training_in_cluster:

            try:
                # "--display" can be used only if "--nv" is defined.
                command = command + " --display" 
                print ( "running \"{0}\"".format(command) )
                subprocess.call ( command, shell=True)
            except:
                print( "Fail to run")

        else:

            # write a sbatch file
            slurm_string = slurm_template( job_name = exp_name, command = command, error= job_errfile, output = job_outfile, **slurm_default )
            filestring = slurm_string.generate_filestring()

            sbatch_file = os.path.join (exp_dir, "experiment.sbatch")
            
            with open ( sbatch_file, "w" ) as f:
                f.write ( filestring )

            # submit the task to the clluster
            try:
                command = "sbatch {0}".format ( sbatch_file ) 
                print ( "running \"{0}\"".format(command) )
                subprocess.call ( command, shell=True)

            except:
                print( "Fail to batch")

if __name__ == '__main__':
    
    main()
