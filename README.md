# build image
docker build -t visual-content-analysis .

# run container
docker run --rm -d -p 80:80 visual-content-analysis

# checkout 
http://localhost:80/docs