#!/bin/zsh

docker build -t controller-image -f Dockerfile.controller .
docker build -t producer-image -f Dockerfile.producer .
docker build -t consumer-image -f Dockerfile.consumer .