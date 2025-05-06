echo "exeucting python on Docker"
docker run --rm --volumes-from=jenkins_save01 ollama
echo "successfully"