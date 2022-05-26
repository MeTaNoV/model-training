docker build -f test/Dockerfile -t tests ../
docker run -u 0 -it tests pytest test/etl

