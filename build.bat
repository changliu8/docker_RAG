echo "exeucting python on Docker"
docker run --rm -w=$WORKSPACE --volumes-from=jenkins_save01 python3.11:haili
echo "successfully"