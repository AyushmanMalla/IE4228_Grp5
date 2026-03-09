

# **A Researcher's Comprehensive Guide to High-Performance Computing on NSCC ASPIRE2A and the Broader GPU Landscape**

## **Section 1: Navigating the ASPIRE2A Supercomputing Ecosystem**

Transitioning from a personal computing environment to a national supercomputing facility represents a significant leap in computational capability. This section provides a foundational understanding of the National Supercomputing Centre's (NSCC) ASPIRE2A system, outlining its architecture, operational principles, and the policies that govern its use. Mastering these core concepts is the first and most critical step toward leveraging this powerful resource for advanced research.

### **1.1 The HPC Paradigm Shift: From Interactive to Batch Processing**

The most fundamental change when moving to a High-Performance Computing (HPC) cluster like ASPIRE2A is the shift from an interactive workflow to a batch-processing model. On a personal computer, commands are executed immediately. A user can launch a computationally intensive script and observe its progress in real-time in the same terminal session. This interactive model is not feasible on a shared supercomputer designed to serve hundreds of researchers simultaneously.

Instead, ASPIRE2A, like all major HPC systems, operates on a managed, scheduled basis. The terminal prompt a user sees upon logging in places them on a **login node**.1 These nodes are shared gateways to the system and are strictly reserved for lightweight, preparatory tasks. These tasks include editing source code and job scripts, compiling programs, managing files, and submitting computational jobs to the workload manager.

Under the NSCC Acceptable Use Policy, it is expressly forbidden to run computationally intensive or long-running processes on the login nodes.1 Doing so would consume shared resources and degrade the interactive experience for all other users. All significant computational work must be encapsulated within a

**job script** and submitted to the system's workload manager, or **scheduler**. The scheduler then places the job in a queue and, when the requested resources become available, executes the script on one or more dedicated **compute nodes**. This "submit-and-wait" model requires a different approach to development. It necessitates careful planning, robust scripting, and an asynchronous method of checking job status and retrieving results once the computation is complete. This batch-processing paradigm is the cornerstone of efficient and fair resource sharing in a large-scale HPC environment.

### **1.2 System Architecture and Hardware Specifications**

ASPIRE2A is a state-of-the-art HPE Cray EX supercomputer, a massively parallel system designed to tackle some of the most demanding computational challenges in science and engineering.2 Its architecture comprises several types of specialized nodes interconnected by a high-speed network, allowing for both single-node and large-scale multi-node parallel jobs.

* **CPU Compute Nodes:** The workhorse of the system for general-purpose scientific computing consists of 768 standard compute nodes. These nodes are equipped with AMD EPYC 7713 processors, each providing 64 physical cores. With hyper-threading, this can present as 128 logical CPUs per node.1 These nodes are suitable for a wide range of applications, from simulations in computational fluid dynamics to large-scale data analysis.  
* **GPU Accelerated Nodes (The AI System):** Of particular interest for modern machine learning and AI workloads is the system's powerful GPU partition. ASPIRE2A is equipped with 82 dedicated GPU nodes, housing a total of 352 NVIDIA A100 Tensor Core GPUs.4 These are the 40 GB SXM4 variant, a high-performance, server-grade GPU specifically designed for scientific computing and deep learning.3 These nodes are the primary resource for training complex neural networks and other GPU-accelerated tasks. They are accessible through a special "AI queue".1 The AI system is further subdivided into nodes containing either four or eight A100 GPUs each, allowing for flexible scaling of GPU resources within a single job.6 It is also worth noting that NSCC has deployed a successor system, ASPIRE2A+, which features even more powerful NVIDIA H100 GPUs, indicating a continued commitment to providing cutting-edge AI hardware.5  
* **Interconnect:** To enable efficient communication between this vast array of nodes, which is critical for large parallel jobs that span multiple nodes, ASPIRE2A uses a high-speed HPE Slingshot-10 interconnect.3 This advanced networking fabric ensures that data can be exchanged between nodes with very low latency and high bandwidth, a prerequisite for achieving performance at scale.

### **1.3 The Filesystem Hierarchy: Where Your Data Lives and Why It Matters**

A common point of confusion and error for new HPC users is the filesystem. Unlike the single, monolithic storage volume of a typical personal computer, HPC systems employ a hierarchical, multi-tiered storage architecture. Each tier is optimized for a different purpose, with distinct characteristics regarding performance, storage capacity (quota), and data longevity (persistence). A proactive data management strategy is therefore not merely a best practice but an essential requirement for successful and reliable computation on ASPIRE2A.

* **/home Directory:** Upon logging in, a user is placed in their home directory, /home/\<userid\>. This is a user's permanent, personal storage space. It is backed up and intended for storing essential files such as source code, configuration files (e.g., .bashrc), SSH keys, and small, critical documents.1 However, the storage quota for  
  /home is relatively small. It is not designed to hold large datasets, software installations, or the input/output files of active computations. Files here are generally safe, though they may be archived if an account is inactive for over 90 days.4  
* **/scratch Directory:** The /scratch filesystem is the primary high-performance, large-capacity workspace for running jobs. It is built on a parallel filesystem (such as Lustre) designed for the high-speed, concurrent I/O operations typical of scientific applications.6 This is the appropriate location for staging large datasets before a job runs and for writing the intermediate and final output files during a job. The critical policy to understand is that  
  /scratch is **temporary, non-archival storage**. Any file within the /scratch directory that has not been accessed for more than 30 days is subject to being automatically deleted by the system's purge policy, without prior notice.9 This policy is necessary to manage the finite high-performance storage space for all users. It is the user's responsibility to copy important results from  
  /scratch to a permanent storage location upon job completion.  
* **/home/project/\<Project ID\> Directory:** For collaborative work and long-term data storage, NSCC provides project directories.1 These are shared spaces accessible by all members of a specific research project. This is the ideal location for storing shared, foundational datasets, project-specific licensed software installations, and important final results that need to be preserved for the duration of the project and beyond. These directories have much larger quotas than  
  /home and are designed for persistence.  
* **/raid Local NVMe Storage:** The nodes in the specialized AI queue are equipped with an additional tier of ultra-high-performance local storage: a Non-Volatile Memory Express (NVMe) solid-state drive mounted at /raid.1 This storage is physically located within the compute node itself, offering the lowest possible latency and highest I/O bandwidth, far exceeding that of the central  
  /scratch filesystem.6 This local storage is specifically intended for the intense, random-access I/O patterns often seen in AI/ML data pipelines (e.g., loading batches of images). However, data on  
  /raid is **ephemeral**; it exists only for the duration of a single job. Any data written to /raid will be lost when the job finishes and the node is allocated to another user. An effective workflow for I/O-bound AI jobs involves staging data from /scratch or /home/project to /raid at the beginning of the job script, performing the computation using the data on /raid, and copying the results back to a persistent location before the job script exits.

This multi-tiered system necessitates a conscious data lifecycle. A robust workflow involves storing master datasets in /home/project, staging the required data subset to /scratch (or /raid for AI jobs) as a first step in a job script, directing the computational job to read from and write to this temporary space, and, as a final step, archiving the critical results back to /home/project.

| Filesystem Path | Purpose | Performance | Persistence Policy | Ideal Use Case |
| :---- | :---- | :---- | :---- | :---- |
| /home/\<userid\> | Permanent user space | Moderate | Permanent (archived after 90 days inactivity) | Source code, configuration files, small documents |
| /scratch/\<userid\> | High-performance job workspace | High (Parallel) | Temporary (files purged after 30 days of no access) | Staging job data, writing computational output |
| /home/project/\<ProjectID\> | Shared, permanent project space | High (Parallel) | Permanent (for project duration) | Shared datasets, licensed software, final results |
| /raid (on AI nodes) | Ultra-fast, job-local storage | Very High (Local NVMe) | Ephemeral (data deleted at end of job) | I/O-intensive AI/ML training data |

### **1.4 User Policies and Getting Help**

All usage of ASPIRE2A is governed by the NSCC Acceptable Use Policy.1 The key principles, as previously mentioned, are that all computational jobs must be run via the scheduler on compute nodes, and users are responsible for the management and security of their own data.

For technical assistance, users should not hesitate to contact the NSCC helpdesk. Problems or questions can be directed via email to help@nscc.sg or submitted as a formal service request through the NSCC Service Desk portal.11 When reporting an issue, it is helpful to provide as much detail as possible, including the user ID, the job ID (if applicable), the exact commands used, and any error messages received.

## **Section 2: Establishing a Secure and Efficient Workflow**

With a conceptual understanding of the ASPIRE2A environment, the next step is to establish a practical, secure, and efficient workflow. This involves configuring access, learning to manage the system's software environment, setting up isolated Python environments for reproducible research, and mastering the methods for transferring data between your local machine and the supercomputer.

### **2.1 Connecting to ASPIRE2A: Your Secure Gateway**

Access to ASPIRE2A is managed through secure protocols and requires authentication tailored to a user's home institution.

* **SSH Protocol and Hostname Selection:** The standard method for accessing a remote Linux system is the Secure Shell (SSH) protocol. The command is ssh \<userid\>@\<hostname\>, where \<userid\> is the assigned 8-character NSCC user ID.4 To ensure optimal network performance, users should connect via the high-speed access link dedicated to their institution. For example, users from the National University of Singapore (NUS) should connect to  
  aspire2a.nus.edu.sg, while users from Nanyang Technological University (NTU) should use aspire2antu.nscc.sg.1 Users not affiliated with a primary stakeholder institution should use the general access point,  
  aspire2a.nscc.sg.4  
* **VPN and Two-Factor Authentication (2FA):** Security is paramount. All users are required to enroll in Duo Two-Factor Authentication (2FA), which adds a critical layer of security to the login process.1 Users from non-stakeholder institutions, or those connecting from outside their campus network, must first establish a connection to the NSCC Virtual Private Network (VPN) before attempting to SSH into the system. Detailed guides for setting up the VPN on Windows, macOS, and Linux are available on the NSCC help portal.1  
* **SSH Keys for Secure, Passwordless Login:** While password-based authentication is supported, the recommended method for both security and convenience is to use SSH keys. This involves generating a cryptographic key pair (a public key and a private key) on the local machine. The public key is uploaded to the NSCC User Portal, while the private key remains securely on the user's computer. Once configured, this allows the user to log in without needing to type a password every time, streamlining the workflow significantly. Instructions for setting up SSH keys can be found in the user portal.10

### **2.2 The Environment Modules System: Taming Software Complexity**

Supercomputers like ASPIRE2A provide a vast and diverse collection of pre-installed scientific software, compilers, and libraries, often with multiple versions of each package available. Managing this complexity and avoiding conflicts is the job of the Environment Modules system. Instead of having all software paths available at all times, the module command allows users to dynamically modify their shell environment for a specific session or job, loading only the software they need.

* **Core Concept:** The module system works by manipulating environment variables such as $PATH (which tells the shell where to find executable programs) and $LD\_LIBRARY\_PATH (which tells the system where to find shared libraries). Loading a module prepends the necessary paths for that software to these variables, making it available to the user. Unloading the module removes them, cleaning the environment.  
* **Essential Commands:** Mastering a few key commands is essential for effective software management on ASPIRE2A:  
  * module avail: This command lists all the software packages and versions available on the system. This is the definitive way to discover what is installed.12  
  * module list: This shows which modules are currently loaded in your active shell session.12  
  * module load \<package\>/\<version\>: This is the primary command to add software to your environment. For example, to use the GCC 11.2.0 compiler and a specific version of Python, one would use module load gcc/11.2.0-nscc python/3.10.4.13 If the version is omitted, the system's default version will be loaded.  
  * module unload \<package\>: This removes a previously loaded module from the environment.  
  * module purge: This command unloads all currently loaded modules, resetting the environment to a clean, default state. This is a crucial command for ensuring reproducibility. It is highly recommended to include module purge at the beginning of every job script to guarantee that the job runs in a predictable environment, unaffected by any modules that might have been loaded in the user's interactive login session.15

The software available includes compilers (GCC, Intel), MPI libraries (Cray-MPICH, OpenMPI), numerical libraries (MKL, FFTW), and a wide range of domain-specific applications and tools like Python, PyTorch, TensorFlow, GROMACS, LAMMPS, CUDA, and cuDNN.12

### **2.3 Python Environment Management with Miniforge3/Conda**

For researchers using Python, managing package dependencies is critical for reproducibility. While the system provides base Python installations via the module system, it is strongly advised to create isolated, project-specific environments to manage libraries. The recommended tool for this on ASPIRE2A is Conda, which is provided through the Miniforge3 module.

Creating isolated environments prevents dependency conflicts between projects and ensures that a project's computational environment can be precisely recreated in the future. This discipline is what elevates a computational workflow from a series of ad-hoc experiments to reproducible science.

* **Step-by-Step Guide to Creating a Conda Environment:**  
  1. **Load the Miniforge3 Module:** Begin by loading the necessary module in your terminal: module load miniforge3/23.10. This makes the conda command available.13  
  2. **Choose a Non-Home Location:** Conda environments can become very large, consuming gigabytes of disk space. The user's /home directory has a small quota and is not suitable for storing them. Attempting to do so can lead to failed package installations or, in severe cases, the inability to log in if the quota is completely filled. Therefore, environments should be created in a location with a larger quota, such as /scratch/\<userid\> or, for long-term projects, /home/project/\<ProjectID\>.18 The project directory is often the superior choice as it is not subject to the 30-day purge policy of  
     /scratch and can be shared with collaborators.  
  3. **Create the Environment:** Use the conda create command with the \--prefix flag to specify the installation path. For example, to create an environment for a machine learning project in the scratch space with Python 3.10 and PyTorch:  
     Bash  
     conda create \--prefix /scratch/\<userid\>/my-ml-env python=3.10 pytorch torchvision \-c pytorch \-c conda-forge

     The \-c flag specifies additional channels (like pytorch and conda-forge) from which to download packages.18  
  4. **Activate and Manage the Environment:** To use the environment, activate it using its full path: conda activate /scratch/\<userid\>/my-ml-env. The command prompt will change to indicate the active environment. Once activated, additional packages can be installed using conda install \<package\> or pip install \<package\>. When finished, the environment can be deactivated with conda deactivate.18  
  5. **Best Practices for Package Installation:** A common source of problems is mixing the conda and pip package managers indiscriminately. They use different dependency resolution mechanisms, which can lead to broken environments. The recommended strategy is to install as many packages as possible with conda in the initial conda create command. If packages must be added later, use conda install first. Only resort to pip for packages that are not available in any Conda channels.18

### **2.4 Data Transfer: Moving Data To and From ASPIRE2A**

An essential part of any computational workflow is moving data—datasets, source code, and results—between a local computer and the HPC system. ASPIRE2A supports several standard, secure methods for file transfer.

* **Command-Line Tools (Recommended):** For Linux, macOS, and modern Windows (via PowerShell or WSL), command-line tools are the most efficient and scriptable methods for data transfer.  
  * **scp (Secure Copy):** This command is suitable for transferring single files or small directories. The syntax is analogous to the standard cp command. For example, to copy a local file to your scratch directory on ASPIRE2A:  
    Bash  
    scp local\_file.zip \<userid\>@aspire2a.nus.edu.sg:/scratch/\<userid\>/

    20  
  * **rsync (Remote Sync):** For transferring large directories, entire datasets, or for workflows that require periodic synchronization, rsync is the superior tool. Its key advantage is its use of a delta-transfer algorithm, which means it only transfers the differences between the source and destination files. This makes it much faster for subsequent transfers. It is also resumable, meaning a transfer that is interrupted by a network issue can be restarted from where it left off without re-transferring the entire dataset. A typical command to synchronize a local data directory with a remote one would be:  
    Bash  
    rsync \-avP local\_dataset/ \<userid\>@aspire2a.nus.edu.sg:/scratch/\<userid\>/remote\_dataset/

    The \-a flag enables archive mode (preserving permissions, etc.), \-v provides verbose output, and \-P shows progress and allows resuming.20  
* **Graphical User Interface (GUI) Tools:** For users who prefer a graphical interface, several clients are available that implement the underlying SSH File Transfer Protocol (SFTP). Popular options include FileZilla (cross-platform), WinSCP (Windows), and Cyberduck (macOS).10 These tools provide a familiar drag-and-drop interface for transferring files. When configuring these clients, the user will need to provide the same hostname, user ID, and credentials (password or SSH private key) used for SSH access, and ensure the port is set to 22\.  
* **Direct Download from the Internet:** For datasets that are publicly available online, it is often most efficient to download them directly to the ASPIRE2A filesystem from a login node. The standard Linux command-line utilities wget and curl are available for this purpose.20

## **Section 3: Mastering the PBS Pro Workload Manager**

The heart of any HPC cluster is its workload manager or scheduler. On ASPIRE2A, this role is filled by PBS Pro (Portable Batch System). Understanding how to communicate with PBS Pro through job scripts is the key to unlocking the computational power of the system's compute nodes. The scheduler acts as a sophisticated gatekeeper, managing access to shared resources like CPUs, GPUs, and memory to ensure fair and efficient utilization for all researchers.

### **3.1 The Role of the Scheduler**

The primary function of the PBS Pro scheduler is to manage a queue of jobs submitted by users. Each job requests a specific set of resources (e.g., a certain number of CPU cores, GPUs, amount of memory, and a maximum run time). The scheduler maintains an inventory of all available resources on the compute nodes and works to fit the queued jobs onto these resources, much like a game of Tetris. Its goal is to maximize the overall system throughput (the number of jobs completed over time) while adhering to site-specific policies regarding job priorities and resource limits.6

This process means that a submitted job will not run immediately unless the requested resources are currently free. It will wait in the queue (Q state) until the scheduler can allocate the necessary resources, at which point it will transition to the running state (R). This negotiation for resources underscores the importance of making accurate requests. A job that requests an entire node for 7 days when it only needs a few cores for 2 hours will likely wait much longer in the queue for that large, long-duration slot to become available. Conversely, underestimating resource needs can cause a job to fail (e.g., by running out of memory) or be prematurely terminated by the system when it exceeds its requested walltime. Therefore, profiling code to understand its requirements and making precise requests is a hallmark of an effective HPC user, leading to faster job turnaround times and better overall system efficiency.

### **3.2 Anatomy of a PBS Job Script**

A PBS job script is a standard shell script (typically Bash) that contains the commands to be executed on a compute node. What makes it a job script are special comment lines at the top of the file, which begin with \#PBS. These lines are not executed by the shell but are interpreted by the qsub command at submission time as directives for the scheduler.6

* **Key PBS Directives:** A well-formed job script must include directives that define the job's name, the project to charge, the queue to use, and, most importantly, the computational resources required.  
  * \#PBS \-N \<JobName\>: Assigns a human-readable name to the job, which is useful for identification in queue listings. Example: \#PBS \-N My\_Simulation\_Run.6  
  * \#PBS \-q \<queue\_name\>: Specifies the destination queue. The primary queues are normal for general CPU and GPU jobs, and the specialized ai queue for jobs requiring access to the NVIDIA A100 nodes.1  
  * \#PBS \-P \<ProjectID\>: Associates the job with a specific project ID, which is used for accounting of computational resource usage.6  
  * \#PBS \-l walltime=HH:MM:SS: Sets the maximum wall-clock time the job is allowed to run. If the job exceeds this limit, it will be automatically terminated by the scheduler. This is a critical parameter for both preventing runaway jobs and allowing the scheduler to plan effectively. Example: \#PBS \-l walltime=24:00:00 for a 24-hour run.6  
  * \#PBS \-j oe: A convenience directive that merges the standard output (stdout) and standard error (stderr) streams into a single output file. This simplifies log checking and debugging.6  
* **Resource Requests (-l select):** The most critical and syntactically complex directive is the resource request line, which uses the select statement. This line tells the scheduler exactly what hardware to allocate.  
  * **CPU Jobs:** The general format is select=\<chunks\>:ncpus=\<cores\>:mem=\<memory\>. A "chunk" typically corresponds to a single compute node. To request one full standard compute node on ASPIRE2A, which has 128 physical cores, the directive would be: \#PBS \-l select=1:ncpus=128:mem=440GB.1 To run a smaller serial job on a single core, one would request:  
    \#PBS \-l select=1:ncpus=1:mem=2GB.1  
  * **GPU Jobs:** To request GPUs, the ngpus attribute is added. A typical single-GPU job request for the ai queue would look like: \#PBS \-l select=1:ncpus=16:ngpus=1. On the AI system, the scheduler is configured to automatically allocate a proportional number of CPU cores and memory along with the GPUs (e.g., 16 CPUs per requested GPU), so specifying a large ncpus value is often unnecessary.6  
  * **Multi-Node and Multi-GPU Jobs:** The select syntax is powerful for scaling up. The distinction between inter-node and intra-node parallelism is important.  
    * \#PBS \-l select=1:ncpus=64:ngpus=4: This requests a single node (select=1) with four GPUs (ngpus=4). This is suitable for applications that can leverage multiple GPUs on the same machine via technologies like NVLink.  
    * \#PBS \-l select=4:ncpus=16:ngpus=1: This requests four separate nodes (select=4), each with one GPU (ngpus=1). This is for distributed parallel applications (e.g., using MPI) that communicate across the network interconnect.6

### **3.3 The Job Lifecycle: Submission, Monitoring, and Management**

Once a job script is written, a set of command-line tools is used to interact with the PBS Pro scheduler throughout the job's lifecycle.

* **Submission:** A job script is submitted to the scheduler using the qsub command. For a script named my\_job.pbs, the command is simply: qsub my\_job.pbs.11 Upon successful submission,  
  qsub will print a unique job ID to the terminal, which is used to track and manage the job. It is also possible to pass variables from the command line into the job script's environment using the \-v flag, which can be useful for parameterizing runs. For example: qsub \-v BATCH\_SIZE=64,LEARNING\_RATE=0.001 my\_job.pbs.13  
* **Monitoring:** Several commands are available to check the status of jobs.  
  * qstat \-u \<userid\>: This is the most common command, listing all jobs owned by the specified user, their current status (e.g., Q for queued, R for running, C for completed), and the queue they are in.  
  * qstat \-f \<jobid\>: This command provides a detailed, comprehensive report on a specific job, including the resources it requested, the resources it is currently using, the node(s) it is running on, and any comments from the scheduler about why it might be queued.6  
  * nqstat: ASPIRE2A also provides nqstat, a more user-friendly tool for viewing the overall status of the queues and jobs on the system.14  
* **Deletion:** If a job needs to be canceled, either while it is queued or already running, the qdel command is used, followed by the job ID: qdel \<jobid\>.

A job script should be treated as more than just a list of commands; it is a self-contained, reproducible scientific instrument. It precisely documents the software environment (via module commands), the hardware resources requested, and the exact command-line arguments used for a given computation. By saving these scripts alongside the source code and results, for example in a Git repository, a researcher creates a complete, executable record of their experiment. This practice is invaluable for ensuring that results can be verified and reproduced months or even years later, which is a fundamental tenet of the scientific method.

| Command/Directive | Function | Example |
| :---- | :---- | :---- |
| **PBS Commands** |  |  |
| qsub \<script\> | Submits a job script to the scheduler. | qsub run\_simulation.pbs |
| qstat \-u \<userid\> | Lists the status of your jobs. | qstat \-u myuser123 |
| qdel \<jobid\> | Deletes/cancels a job. | qdel 1234567.pbs101 |
| qstat \-f \<jobid\> | Displays detailed information about a job. | qstat \-f 1234567.pbs101 |
| **PBS Directives** |  |  |
| \#PBS \-N \<name\> | Sets the job name. | \#PBS \-N ProteinFolding\_Run3 |
| \#PBS \-q \<queue\> | Specifies the job queue. | \#PBS \-q ai |
| \#PBS \-P \<project\> | Sets the project ID for accounting. | \#PBS \-P 12345678 |
| \#PBS \-l walltime=T | Requests maximum wall-clock time. | \#PBS \-l walltime=08:00:00 |
| \#PBS \-j oe | Merges stdout and stderr into one file. | \#PBS \-j oe |
| \#PBS \-l select=... | Specifies resource request (nodes, CPUs, GPUs). | \#PBS \-l select=1:ncpus=16:ngpus=1 |

## **Section 4: A Practical Guide to GPU-Accelerated AI/ML on ASPIRE2A**

This section synthesizes the concepts from the preceding sections into a practical, end-to-end walkthrough of a typical GPU-accelerated machine learning workflow on ASPIRE2A. The example will cover accessing the AI system, using containers for a reproducible environment, and the complete lifecycle of a deep learning training job from data preparation to results archival.

### **4.1 Accessing the AI System and NVIDIA A100 GPUs**

The powerful NVIDIA A100 GPUs on ASPIRE2A are a specialized resource, accessible only through a dedicated configuration.

* **The ai Queue:** All jobs that require GPUs must be submitted to the ai queue. This is specified in the job script with the directive \#PBS \-q ai.1 Access to this queue is not enabled by default for all users and may need to be specifically requested depending on project requirements.  
* **Requesting GPUs:** The number of GPUs is requested using the ngpus parameter within the \-l select directive. For a typical job that uses a single GPU on one node, the resource request line is \#PBS \-l select=1:ngpus=1.6 As noted previously, the scheduler on the AI system will automatically co-allocate an appropriate number of CPU cores and memory with the GPU request. To scale up, a user can request multiple GPUs on a single node (e.g.,  
  \#PBS \-l select=1:ngpus=4) for data-parallel training on one machine, or request GPUs across multiple nodes (e.g., \#PBS \-l select=4:ngpus=1) for more complex distributed training models.6  
* **Verifying GPU Allocation:** Once a job is running on a compute node, it is good practice to verify that the GPU resources have been allocated correctly. This can be done by including the nvidia-smi command in the job script. This command provides a detailed status of the NVIDIA GPUs on the node. Because PBS uses control groups (cgroups) for resource isolation, the job's environment will be restricted to only see and use the specific GPUs that were allocated to it.6 The scheduler also sets the  
  $CUDA\_VISIBLE\_DEVICES environment variable, which informs CUDA-aware applications which GPU devices they are permitted to use.6 This ensures that multiple independent GPU jobs can run on the same physical node without interfering with one another.

### **4.2 Leveraging Containers with Singularity for Reproducible AI Environments**

While Conda provides excellent environment management for Python packages, an even more robust and portable solution for reproducibility is containerization. NSCC provides the Singularity container platform, which allows an entire software environment—including the operating system, system libraries, programming languages, and specific versions of frameworks like PyTorch and TensorFlow—to be packaged into a single, immutable file called a container image.

Using pre-built, optimized containers provided by NSCC is highly recommended. This approach eliminates the complex and error-prone process of manually installing deep learning frameworks and ensuring their compatibility with the system's specific NVIDIA drivers and CUDA libraries. The combination of the specialized ai queue, the high-performance local /raid storage, and curated Singularity containers effectively creates a highly optimized "AI/ML appliance" within the broader supercomputer. The most efficient path to high performance is to embrace this integrated system.

* **Using a Pre-built PyTorch Container:** To use a container within a job, the script must invoke the singularity command.  
  * First, the path to the desired container image is typically defined in a shell variable: image="/app/apps/containers/pytorch/pytorch-nvidia-22.02-py3.sif".6  
  * Next, the singularity run command is used to execute a command inside the container. The \--nv flag is absolutely essential; it instructs Singularity to bind the host system's NVIDIA drivers and GPUs into the container, enabling GPU acceleration. Without this flag, the container would not be able to access the A100 GPUs. The full command would look like: singularity run \--nv $image python my\_training\_script.py.6

### **4.3 End-to-End Workflow Example: Training a Deep Learning Model**

This example demonstrates the complete process of training a simple convolutional neural network on the CIFAR-10 dataset using PyTorch inside a Singularity container.

* **Step 1: Data and Code Preparation (on a Login Node)**  
  1. Log in to ASPIRE2A and navigate to your scratch directory: cd /scratch/\<userid\>.  
  2. Create a project directory: mkdir cifar10-project && cd cifar10-project.  
  3. Download the CIFAR-10 dataset. While PyTorch can download it automatically, for larger datasets it is best practice to download them manually first: wget https://www.cs.toronto.edu/\~kriz/cifar-10-python.tar.gz followed by tar \-xvzf cifar-10-python.tar.gz. This creates a cifar-10-batches-py directory.  
  4. Using a text editor (like nano or vim), create the Python training script, train\_cifar.py. This script would define the neural network, the data loading logic, the training loop, and would save the trained model's weights to a file.  
* Step 2: Writing the PBS Job Script (run\_cifar\_training.pbs)  
  Create the job script that orchestrates the entire process. This script combines PBS directives, environment setup, and the execution command.  
  Bash  
  \#\!/bin/bash  
  \#PBS \-N Pytorch\_CIFAR10  
  \#PBS \-q ai  
  \#PBS \-P \<YourProjectID\>  
  \#PBS \-l walltime=00:30:00  
  \#PBS \-l select=1:ncpus=16:ngpus=1  
  \#PBS \-j oe

  \#\#\# \--- Environment Setup \---  
  echo "Job started on $(hostname) at $(date)"

  \# Change to the directory where the job was submitted from.  
  \# This is a crucial step to ensure the script finds the data and code.  
  cd "$PBS\_O\_WORKDIR" |

| exit 1

\# Optional: Verify GPU allocation  
echo "Checking GPU allocation:"  
nvidia-smi  
echo "CUDA\_VISIBLE\_DEVICES \= $CUDA\_VISIBLE\_DEVICES"

\#\#\# \--- Execution \---  
echo "Starting PyTorch training inside Singularity container..."

\# Define the path to the NSCC-provided PyTorch container  
IMAGE\_PATH="/app/apps/containers/pytorch/pytorch-nvidia-22.02-py3.sif"

\# Execute the training script inside the container.  
\# The \--nv flag is essential for enabling GPU access.  
singularity run \--nv "$IMAGE\_PATH" python train\_cifar.py \--data-path./cifar-10-batches-py

echo "Training finished at $(date)."  
\#\#\# \--- Job End \---  
\`\`\`

* **Step 3: Submitting and Monitoring the Job**  
  1. From the terminal in the cifar10-project directory, submit the job: qsub run\_cifar\_training.pbs. Note the job ID that is returned.  
  2. Monitor the job's status: qstat \-u \<userid\>. Wait for the status to change from Q to R.  
  3. While the job is running, you can check the contents of the output file in real-time to see the print statements from your script: tail \-f Pytorch\_CIFAR10.o\<jobid\>.  
* **Step 4: Retrieving and Archiving Results**  
  1. Once the job completes, the output file will contain the full log of the run, including the output from nvidia-smi and any accuracy/loss metrics printed by the Python script.  
  2. The training script will have saved the model weights (e.g., cifar\_net.pth) in the same directory.  
  3. This is the most critical final step: archive the important results. The code, the final model, and the log file should be copied from the temporary /scratch space to your permanent /home/project directory to prevent them from being deleted by the purge policy.  
     Bash  
     \# Create a results directory in your project space  
     mkdir \-p /home/project/\<ProjectID\>/cifar10-results/run-$(date \+%F-%H%M)

     \# Copy the essential files for archival  
     cp train\_cifar.py run\_cifar\_training.pbs Pytorch\_CIFAR10.o\<jobid\> cifar\_net.pth /home/project/\<ProjectID\>/cifar10-results/run-$(date \+%F-%H%M)/

This disciplined workflow ensures that every experiment is reproducible, its results are safely stored, and the powerful GPU resources of ASPIRE2A are used effectively.
