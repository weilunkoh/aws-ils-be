ECHO "Building aws-ils-be version 1.1.0"
docker build -t mitb/aws-ils-be:1.1.0 .
docker save -o deployment/mitb-aws-ils-be.tar.gz mitb/aws-ils-be:1.1.0