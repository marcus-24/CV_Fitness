OPENPOSE_URL="https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models/"
POSE_FOLDER="pose/"

MPI_FOLDER=${POSE_FOLDER}"mpi/"
MPI_MODEL=${MPI_FOLDER}"pose_iter_160000.caffemodel"
wget -c ${OPENPOSE_URL}${MPI_MODEL} -P ${MPI_FOLDER}
