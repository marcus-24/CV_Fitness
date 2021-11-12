@echo off
echo ----------Downloading body pose model----------
SET OPENPOSE_URL=http://posefs1.perception.cs.cmu.edu/OpenPose/models/
SET POSE_FOLDER=pose/

SET MPI_FOLDER=%POSE_FOLDER%mpi/
SET MPI_MODEL=%MPI_FOLDER%pose_iter_160000.caffemodel
curl %OPENPOSE_URL%%MPI_MODEL% --output %MPI_MODEL%

SET ARCH_URL=https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/
SET ARCH_FILE=%MPI_FOLDER%pose_deploy_linevec.prototxt
curl %ARCH_URL%%ARCH_FILE% --output %ARCH_FILE%
