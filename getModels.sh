OPENPOSE_URL="http://posefs1.perception.cs.cmu.edu/OpenPose/models/"
POSE_FOLDER="pose/"

MPI_FOLDER=${POSE_FOLDER}"mpi/"
MPI_MODEL=${MPI_FOLDER}"pose_iter_160000.caffemodel"
wget -c ${OPENPOSE_URL}${MPI_MODEL} -P ${MPI_FOLDER}

ARCH_URL="https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/"
ARCH_FILE=${MPI_FOLDER}"pose_deploy_linevec.prototxt"
wget ${ARCH_URL}${ARCH_FILE} -P ${MPI_FOLDER}
