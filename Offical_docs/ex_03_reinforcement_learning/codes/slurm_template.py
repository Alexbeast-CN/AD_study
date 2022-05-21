

class slurm_template:

    def __init__(self, job_name = 'experiment', partition="test", cpus_per_task = 1, mem_per_cpu = "2G",
                gres = "gpu:1", time = "24:00:00", error = "job.%J.err", output = "job.%J.out",
                 mail_type ="ALL", mail_user="", command = ""):

        self.job_name = job_name
        self.partition = partition
        self.cpus_per_task = cpus_per_task
        self.mem_per_cpu = mem_per_cpu
        self.gres = gres
        self.time = time
        self.error_file = error
        self.out_file = output
        self.mail_type = mail_type
        self.mail_user = mail_user
        self.precommand = ""
        self.command = command

    def generate_filestring(self):

        return """#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name={0}

#resources:

#SBATCH --cpus-per-task={1:d}
# the job can use and see {1:d} CPUs (from max 24).

#SBATCH --partition={2}
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu={3}
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres={4}
#the job can use and see 1 GPUs (4 GPUs are available in total on one node) use SBATCH --gres=gpu:1080ti:1 to explicitly demand a Geforce 1080 Ti GPU. Use SBATCH --gres=gpu:A4000:1 to explicitly demand a RTX A4000 GPU 

#SBATCH --time={5}
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error={6}
# write the error output to job.*jobID*.err

#SBATCH --output={7}
# write the standard output to job.out

#SBATCH --mail-type={8}
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user={9}
# your mail address

{10}

####
#c) Execute your code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
####

{11}

echo *jobID* DONE!
""".format ( self.job_name, self.cpus_per_task, self.partition, self.mem_per_cpu, self.gres, 
        self.time, self.error_file, self.out_file, self.mail_type, self.mail_user, self.precommand, self.command)
