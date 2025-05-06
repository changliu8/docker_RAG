echo "exeucting python on Docker"
docker run --rm --volumes-from=jenkins_save01 python3.11:haili
echo "successfully"