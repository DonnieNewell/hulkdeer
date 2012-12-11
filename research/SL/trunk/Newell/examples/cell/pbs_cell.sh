#!/bin/sh
###############################################################
#                                                             #
#    Bourne shell script for submitting a parallel MPICH job  #
#    to the PBS queue using the qsub command.                 #
#                                                             #
###############################################################

#     Remarks: A line beginning with # is a comment.
#        A line beginning with #PBS is a PBS directive.
#              PBS directives must come first; any directives
#                 after the first executable statement are ignored.
#
   
##########################
#                        #
#   The PBS directives   #
#                        #
##########################

#          Set the name of the job (up to 15 characters, 
#          no blank spaces, start with alphanumeric character)

#PBS -N Cell

#           Request how many NVIDIA GPU nodes you want

# #PBS -l cuda_gpu=1

#          Specify the number of nodes requested and the
#          number of processors per node. 

# #PBS -l nodes=6:ppn=1
#PBS -l nodes=cray1+cray6+cray7+cray8
#          By default, the standard output and error streams are sent
#          to files in the current working directory with names:
#              job_name.osequence_number  <-  output stream
#              job_name.esequence_number  <-  error stream
#          where job_name is the name of the job and sequence_number 
#          is the job number assigned when the job is submitted.
#          Use the directives below to change the files to which the
#          standard output and error streams are sent.

#    #PBS -o stdout_file
#    #PBS -e stderr_file

#          The directive below directs that the standard output and
#          error streams are to be merged, intermixed, as standard
#          output. 

#PBS -j oe

#          Specify the name of the file standard out will be redirected
#          into.

#PBS -o output1

#          Specify the maximum cpu and wall clock time. The wall
#          clock time should take possible queue waiting time into
#          account.  Format:   hhhh:mm:ss   hours:minutes:seconds
#          Be sure to specify a reasonable value here.
#          If the job does not finish by the time reached,
#          the job is terminated.

# #PBS -l     cput=0:10:00
# #PBS -l walltime=24:00:00

#          export all of my environment variables to the job.

#PBS -V

#          Specify the queue.  The UVA CS dept has: generals. 

#PBS -q generals

#          Specify the maximum amount of physical memory required per process.
#          kb for kilobytes, mb for megabytes, gb for gigabytes.
#          Take some care in setting this value.  Setting it too large
#          can result in your job waiting in the queue for sufficient
#          resources to become available.

#PBS -l mem=12gb

#          PBS can send informative email messages to you about the
#          status of your job.  Specify a string which consists of
#          either the single character "n" (no mail), or one or more
#          of the characters "a" (send mail when job is aborted),
#          "b" (send mail when job begins), and "e" (send mail when
#          job terminates).  The default is "a" if not specified.
#          You should also specify the email address to which the
#          message should be send via the -M option.

#PBS -m abe
#  #PBS -m ae

#PBS -M den4gr@virginia.edu

#          Declare the time after which the job is eligible for execution.
#          If you wish the job to be immediately eligible for execution,
#          comment out this directive.  If you wish to run at some time in 
#          future, the date-time argument format is
#                      [DD]hhmm
#          If the day DD is not specified, it will default to today if the
#          time hhmm is in the future, otherwise, it defaults to tomorrow.
#          If the day DD is specified as in the future, it defaults to the
#          current month, otherwise, it defaults to next month.

# #PBS -a 2215  commented out

#          Specify the priority for the job.  The priority argument must be
#          an integer between -1024 and +1023 inclusive.  The default is 0.

#  #PBS -p 0

#          Define the interval at which the job will be checkpointed,
#          if checkpointing is desired, in terms of an integer number
#          of minutes of CPU time.

#  #PBS -c c=2

##########################################
#                                        #
#   Output some useful job information.  #
#                                        #
##########################################

NCPU=`wc -l < $PBS_NODEFILE`
NNODES=`uniq $PBS_NODEFILE | wc -l`

echo ------------------------------------------------------
echo ' This job is allocated on '${NCPU}' cpu(s)'
echo 'Job is running on node(s): '
cat $PBS_NODEFILE
echo ------------------------------------------------------
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: originating queue is $PBS_O_QUEUE
echo PBS: executing queue is $PBS_QUEUE
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: execution mode is $PBS_ENVIRONMENT
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: node file is $PBS_NODEFILE
echo PBS: number of nodes is $NNODES
echo PBS: current home directory is $PBS_O_HOME
echo PBS: current temporary directory is $TMPDIR
echo PBS: PATH = $PBS_O_PATH
echo ------------------------------------------------------

##############################################################
#                                                            #
#   The prologue script automatically makes a directory      #
#   on the local disks for you.  The name of this directory  #
#   depends on the job id, but you need only refer to it     #
#   using ${WORKDIR}.                                        #
#                                                            #
##############################################################

SERVER=$PBS_O_HOST
WORKDIR=$TMPDIR
SCP=/usr/bin/scp
SSH=/usr/bin/ssh
RSYNC="/usr/bin/rsync -ave ssh"
MACHINES=$PBS_NODEFILE
PATH=$PATH:/usr/bin ; export PATH

######################################################################
#                                                                    #
#   To minimize communications traffic, it is best for your job      #
#   to work with files on the local disk of the compute node.        #
#   Hence, one needs to transfer files from your permanent home      #
#   directory tree to the directory ${WORKDIR} automatically         #
#   created by PBS on the local disk before program execution,       #
#   and to transfer any important output files from the local        #
#   disk back to the permanent home directory tree after program     #
#   execution is completed.                                          #
#                                                                    #
#   There are essentially two ways to achieve this: (1) to use the   #
#   PBS stagein and stageout utilities, or (2) to manually copy the  #
#   files by commands in this script.  The stagein and stageout      #
#   features of OpenPBS are somewhat awkward, especially since       #
#   wildcards and macros in the file lists cannot be used.  This     #
#   method also has some timing issues.  Hence, we ask you to use    #
#   the second method, and to use secure copy (scp) to do the file   #
#   transfers to avoid NSF bottlenecks.                              #
#                                                                    #
######################################################################

#####################################################
#                                                   #
#    Specify the permanent directory(ies) on the    #
#    server host.  Note that when the job begins    #
#    execution, the current working directory at    #
#    the time the qsub command was issued becomes   #
#    the current working directory of the job.      #
#                                                   #
#####################################################


PERMDIR=${HOME}/hulkdeer/research/SL/trunk/Newell/examples/cell

SERVPERMDIR=${PBS_O_HOST}:${PERMDIR}

echo server is $SERVER
echo workdir is $WORKDIR
echo permdir is $PERMDIR
echo servpermdir is $SERVPERMDIR
echo ------------------------------------------------------
echo 'Job is running on node(s): '
cat $PBS_NODEFILE
echo ------------------------------------------------------
echo ' '
echo ' '

###############################################################
#                                                             #
#    Transfer files from server to local disks.               #
#    Start up the mpd daemons and check the mpd ring.         #
#                                                             #
###############################################################

stagein()
{
 if [ -r $MACHINES ] ; then
    machines=$(sort $MACHINES | uniq )
 else
    machines=$(hostname)
 fi
 for machine in $machines ; do

    echo ' '
    echo checking compute node ${machine}
    echo Writing files in node directory ${WORKDIR}
    ${SSH} ${machine} mkdir -p ${WORKDIR}
    ${SCP} ${SERVPERMDIR}/distributedCell ${machine}:${WORKDIR}
    #${SCP} ${SERVPERMDIR}/input_file ${machine}:${WORKDIR}

    echo Files in node work directory are as follows:
    ${SSH} ${machine} ls -l ${WORKDIR}
    
 done

 #echo ' '
 #echo ' starting up mpd daemons '
 #export MPD_CON_EXT=${PBS_JOBID}
 #mpdboot -n ${NNODES} -f ${PBS_NODEFILE} -v --remcons
 #sleep 10
 #mpdtrace -l
 #mpdringtest 100

}

############################################################
#                                                          #
#    Execute the run.  Do not run in the background.       #
#                                                          #
############################################################

runprogram()
{
#cd ${WORKDIR}
  cd $PBS_O_WORKDIR
  for DATA_WIDTH in 512 640
  do
    ITERATIONS=32
    PYRAMID_HEIGHT=1
    CHUNKS_PER_DIMENSION=16
    for LOAD_BALANCE in 0 1
    do
      for PROCESSOR_CONFIG in 1 2 3
      do
        EXEC_FLAGS="$DATA_WIDTH $DATA_WIDTH $DATA_WIDTH $ITERATIONS \
                    $PYRAMID_HEIGHT $CHUNKS_PER_DIMENSION $LOAD_BALANCE \
                    $PROCESSOR_CONFIG"
        #MPI_FLAGS="-verbose"
#MPI_FLAGS="-pernode -verbose"
        LAUNCH="mpiexec -n $NNODES"
        echo "$LAUNCH $MPI_FLAGS ./distributedCell $EXEC_FLAGS"
        $LAUNCH $MPI_FLAGS ./distributedCell $EXEC_FLAGS
      done
    done
  done
}

###########################################################
#                                                         #
#   Copy necessary files back to permanent directory.     #
#   Kill the mdp daemons.                                 #
#                                                         #
###########################################################

stageout()
{
  echo ' '
#    echo 'Killing mpd daemons'
#    mpdallexit

    echo ' '
    echo Transferring files from compute nodes to server
    echo Writing files in permanent directory  ${PERMDIR}
  cd ${WORKDIR}
  OUTFILE="output${PBS_JOBID}"
    ${SCP} output  ${SERVPERMDIR}/${OUTFILE}

  echo Final files in permanent data directory:
    ${SSH} ${SERVER} "cd ${PERMDIR}; ls -l"
}

#####################################################################
#                                                                   #
#  The "qdel" command is used to kill a running job.  It first      #
#  sends a SIGTERM signal, then after a delay (specified by the     #
#  "kill_delay" queue attribute (set to 60 seconds), unless         #
#  overridden by the -W option of "qdel"), it sends a SIGKILL       #
#  signal which eradicates the job.  During the time between the    #
#  SIGTERM and SIGKILL signals, the "cleanup" function below is     #
#  run. You should include in this function commands to copy files  #
#  from the local disk back to your home directory.  Note: if you   #
#  need to transfer very large files which make take longer than    #
#  60 seconds, be sure to use the -W option of qdel.                #
#                                                                   #
#####################################################################

early()
{
  echo ' '
    echo ' ############ WARNING:  EARLY TERMINATION #############'
    echo ' '
}

trap 'early; stageout' 2 9 15


##################################################
#                                                #
#   Staging in, running the job, and staging out #
#   were specified above as functions.  Now      #
#   call these functions to perform the actual   #
#   file transfers and program execution.        #
#                                                #
##################################################

#stagein
runprogram
#stageout 

###############################################################
#                                                             #
#   The epilogue script automatically deletes the directory   #
#   created on the local disk (including all files contained  #
#   therein.                                                  #
#                                                             #
###############################################################

    exit
