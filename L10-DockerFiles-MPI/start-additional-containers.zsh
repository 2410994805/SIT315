#!/bin/zsh

# Check if the number of containers to start is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <number_of_containers>"
  exit 1
fi

# Number of containers to start
NUM_CONTAINERS=$1

# Image names
CONSUMER_IMAGE="consumer-image"
PRODUCER_IMAGE="producer-image"

# Start controller container
echo "Starting controller-container from image $CONSUMER_IMAGE"
# docker run -d --name "controller-container" "$CONSUMER_IMAGE"

# Start consumer and producer containers
for (( i=1; i<=NUM_CONTAINERS; i++ )); do
  # Start producer containers
  PRODUCER_NAME="producer-container$i"
  echo "Starting $PRODUCER_NAME from image $PRODUCER_IMAGE"
  docker run -d --network sit315-network --name "$PRODUCER_NAME" "$PRODUCER_IMAGE"
  
  # Start consumer containers
  CONSUMER_NAME="consumer-container$i"
  echo "Starting $CONSUMER_NAME from image $CONSUMER_IMAGE"
  docker run -d --network sit315-network --name "$CONSUMER_NAME" "$CONSUMER_IMAGE"
done

echo "Started 1 controller container, $NUM_CONTAINERS producer containers, and $NUM_CONTAINERS consumer containers."
