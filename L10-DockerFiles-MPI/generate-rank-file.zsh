#!/bin/zsh

# Check if the number of containers to generate is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <number_of_containers>"
  exit 1
fi

# Number of containers to generate
NUM_CONTAINERS=$1

# Output file
OUTPUT_FILE="rankfile.txt"

# Start with rank 0
RANK=0

# Write rank 0 for controller-container
echo "rank $RANK=controller-container slots=1" > "$OUTPUT_FILE"

# Start from rank 1
((RANK++))

# Variables to track container numbers
PRODUCER_COUNT=1
CONSUMER_COUNT=1

# Loop to generate the rankfile content
for (( i=1; i<=NUM_CONTAINERS; i++ )); do
  if (( i % 2 == 1 )); then
    # Odd indices for producer containers
    CONTAINER_NAME="producer-container$PRODUCER_COUNT"
    ((PRODUCER_COUNT++))
  else
    # Even indices for consumer containers
    CONTAINER_NAME="consumer-container$CONSUMER_COUNT"
    ((CONSUMER_COUNT++))
  fi

  # Write to the rankfile
  echo "rank $RANK=$CONTAINER_NAME slots=1" >> "$OUTPUT_FILE"

  # Increment rank
  ((RANK++))
done

echo "Generated rankfile with ranks from 0 to $((RANK - 1)) in $OUTPUT_FILE."
