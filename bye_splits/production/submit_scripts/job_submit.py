#!/usr/bin/env python

import os
import sys
from datetime import datetime

parent_dir = os.path.abspath(__file__ + 5 * "../")
sys.path.insert(0, parent_dir)

from bye_splits.utils import params, common

import subprocess
import re
import yaml

# Read particle specific variables from the YAML file
particle_var = lambda part, var: config["job"][part][var]

class JobBatches:
    def __init__(self, particle, config):
        self.particle = particle
        self.config = config

    def setup_batches(self):

        my_batches = lambda files, files_per_batch: [files[i: i + files_per_batch] for i in range(0, len(files), files_per_batch)]
        strip_trail = lambda batches: [[file.rstrip() for file in batch] for batch in batches]

        read_dir = self.config["job"]["read_dir"]
        files = particle_var(self.particle, "files")
        files_per_batch = particle_var(self.particle, "files_per_batch")
        if not read_dir:
            with open(files, "r") as File:
                lines = File.readlines()
                # readlines() keeps the explicit "/n" character, strip_trail removes this
                batches = strip_trail(my_batches(lines, files_per_batch))
        else:
            part_submit_dir = particle_var(self.particle, "submit_dir") + "ntuples/"
            paths = [
                "{}{}".format(part_submit_dir, file) for file in os.listdir(part_submit_dir) if file.startswith("ntuple")
            ]
            batches = my_batches(paths, files_per_batch)

        return batches
    
class CondJobBase(JobBatches):
    def __init__(self, particle, config):
        super().__init__(particle, config)
        self.particle_dir = particle_var(self.particle, "submit_dir")
        self.script = config["job"]["script"]
        self.queue = config["job"]["queue"]
        self.proxy = config["job"]["proxy"]
        self.local = config["job"]["local"]
        self.user = config["job"]["user"]


    def write_batch_files(self):
        batch_dir = "{}batches/".format(self.particle_dir)
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
        batch_script_dir = "{}{}/".format(batch_dir, os.path.splitext(os.path.basename(self.script))[0])
        if not os.path.exists(batch_script_dir):
            os.makedirs(batch_script_dir)

        batches = self.setup_batches()
        for i, batch in enumerate(batches):
            out_name = "{}batch_{}.txt".format(batch_script_dir, i)
            if not os.path.exists(out_name):
                with open(out_name, "w") as batch_file:
                    for path in batch:
                        batch_file.write("{}\n".format(path))


    def prepare_batch_submission(self):
        _, script_ext = os.path.splitext(self.script)

        sub_dir = "{}subs/".format(self.particle_dir)
        eos_home = "/eos/user/{}/{}/".format(self.user[0], self.user)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        script_basename = os.path.basename(self.script).replace(".sh", "").replace(".py", "")

        submit_file_name_template = "{}{}_submit.sh".format(sub_dir, script_basename)

        submit_file_name = common.increment_version(submit_file_name_template)
        with open(submit_file_name, "w") as submit_file:
            submit_file.write("#!/usr/bin/env bash\n")
            submit_file.write("workdir={}/bye_splits/production/submit_scripts\n".format(parent_dir))
            submit_file.write("cd $workdir\n")
            submit_file.write("export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch\n")
            submit_file.write("export SITECONFIG_PATH=$VO_CMS_SW_DIR/SITECONF/T2_FR_GRIF_LLR/GRIF-LLR/\n")
            submit_file.write("source $VO_CMS_SW_DIR/cmsset_default.sh\n")
            if script_ext == ".sh":
                submit_file.write("bash {} $1".format(self.script))
            elif script_ext == ".py":
                submit_file.write("python {} --batch_file $1 --user {}".format(self.script, self.user))
        st = os.stat(submit_file_name)
        os.chmod(submit_file_name, st.st_mode | 0o744)


    def prepare_multi_job_condor(self):
        log_dir = "{}logs/".format(self.particle_dir)
        batch_dir = "{}batches/".format(self.particle_dir)
        batch_script_dir = "{}{}/".format(batch_dir, os.path.splitext(os.path.basename(self.script))[0])

        batch_files = os.listdir(batch_script_dir)
        batch_files = [file for file in batch_files if file.endswith(".txt")]

        script_basename = os.path.basename(self.script).replace(".sh", "").replace(".py", "")

        sub_file = "{}subs/{}_submit.sh".format(self.particle_dir, script_basename)
        job_file_name_template = "{}jobs/{}.sub".format(self.particle_dir, script_basename)

        sub_file = common.grab_most_recent(sub_file)
        job_file_name = common.increment_version(job_file_name_template)

        if os.path.exists(job_file_name):
            os.remove(job_file_name)
        with open(job_file_name, "w") as job_file:
            job_file.write("executable = {}\n".format(sub_file))
            job_file.write("Universe              = vanilla\n")
            job_file.write("Arguments             = $(filename)\n")
            job_file.write("output = {}{}_C$(Cluster)P$(Process).out\n".format(log_dir, script_basename))
            job_file.write("error = {}{}_C$(Cluster)P$(Process).err\n".format(log_dir, script_basename))
            job_file.write("log = {}{}_C$(Cluster)P$(Process).log\n".format(log_dir, script_basename))
            job_file.write("getenv                = true\n")
            job_file.write("T3Queue = {}\n".format(self.queue))
            job_file.write("WNTag                 = el7\n")
            job_file.write('+SingularityCmd       = ""\n')
            job_file.write("include: /opt/exp_soft/cms/t3/t3queue |\n")
            job_file.write("queue filename from (\n")
            for file in batch_files:
                job_file.write("{}{}\n".format(batch_script_dir, file))
            job_file.write(")")

        st = os.stat(job_file_name)
        os.chmod(job_file_name, st.st_mode | 0o744)


class CondJob(CondJobBase):
    def __init__(self, particle, config):
        super().__init__(particle, config)


    def prepare_jobs(self):
        self.write_batch_files()

        configs = lambda dir: dir + "configs"
        jobs = lambda dir: dir + "jobs"
        logs = lambda dir: dir + "logs"

        config_dir = configs(self.particle_dir)
        job_dir = jobs(self.particle_dir)
        log_dir = logs(self.particle_dir)

        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        if not os.path.exists(job_dir):
            os.makedirs(job_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.prepare_batch_submission()
        self.prepare_multi_job_condor()


    def launch_job(self):

        if self.local == True:
            machine = "local"
        else:
            machine = "llrt3.in2p3.fr"

        sub_comm = ["condor_submit"]

        script_basename = os.path.basename(self.script).replace(".sh", "").replace(".py", "")

        if not self.local:
            print(
                "\nSending {} jobs on {}".format(self.particle, self.queue + "@{}".format(machine))
            )
            print("===============")
            print("\n")

        sub_args = []
        sub_file_name = "{}jobs/{}.sub".format(self.particle_dir, script_basename)
        sub_file_name = common.grab_most_recent(sub_file_name)
        sub_args.append(sub_file_name)

        if self.local:
            comm = sub_args
        else:
            comm = sub_comm + sub_args

        print(str(datetime.now()), " ".join(comm))
        status = subprocess.run(comm)

if __name__ == "__main__":    
    with open(params.CfgPath, "r") as afile:
        config = yaml.safe_load(afile)

    #for particle in ("photons", "electrons", "pions"):
    for particle in ("photons", "electrons"):
        job = CondJob(particle, config)
        job.prepare_jobs()
        job.launch_job()