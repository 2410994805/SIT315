#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <tuple>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <iomanip>

bool isPalindrome(const std::vector<int> &arr)
{
    // Recognizing Palindrawesome yet? :D
    for (int start = 0, end = arr.size() - 1; start < end; start++, end--)
        if (arr[start] != arr[end])
            return false;

    return true;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // std::cout << "Processor Rank: " << rank << std::endl;

    int numProcesses;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    char *roleEnv = std::getenv("ROLE");
    std::string role = (roleEnv != nullptr) ? std::string(roleEnv) : "unknown";

    if (numProcesses < 3 || numProcesses % 2 == 0)
    {
        std::cerr << "This program requires at least 3 processes and number of processes should be odd." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int controllerRank = 0;
    // Producer ranks will be odd, while consumer ranks will be even
    // const int producerRank = 1;
    // const int consumerRank = 2;

    const int abortSignalTag = 2;
    bool abortSignalReceived = false;

    int arraySize = 16; // Default array size
    if (argc > 1)
    {
        arraySize = std::atoi(argv[1]);
    }

    // Each producer/consumer process will receive an equal amount of elements. E.g. 27 processes: 1 controller, 13 producer slots, 13 consumer slots
    int elementsPerProcess = arraySize / ((numProcesses - 1) / 2);
    int numTuples = elementsPerProcess / 2; // rewrite making it dynamic (MS)

    // std::vector<int> array(elementsPerProcess);

    if (rank == controllerRank)
    {
        uint32_t x = 1;
        char *c = reinterpret_cast<char *>(&x);
        if (*c)
        {
            std::cout << "Little Endian" << std::endl;
        }
        else
        {
            std::cout << "Big Endian" << std::endl;
        }
        std::cout << "Number of processes: " << numProcesses << std::endl;
        std::cout << "Elements per Process: " << elementsPerProcess << std::endl;
        std::cout << "Controller " << rank << " started" << std::endl;
        bool abortSignalSent = false;
        int abortSignal = 0;
        int isPalindrome = 1; // 1 represents a valid palindrome initially

        // Receiving data from consumers and aborting if palindrome false is found
        for (int consumerRank = 2; consumerRank < numProcesses; consumerRank += 2)
        {
            int consumerResult;
            MPI_Recv(&consumerResult, 1, MPI_INT, consumerRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Controller" << rank << " received data from Consumer" << consumerRank << ": " << consumerResult << std::endl;

            if (consumerResult == 0)
            { // Example condition to trigger abort
                std::cout << "Consumer " << consumerRank << " found the broken palindrome condition" << std::endl;
                abortSignal = 1;
                abortSignalSent = true;
                break;
            }
        }

        // "Broadcast" abort signal to all producers
        if (abortSignalSent)
        {
            isPalindrome = 0;
            for (int i = 1; i < numProcesses; i += 2)
            {
                MPI_Send(&abortSignal, 1, MPI_INT, i, abortSignalTag, MPI_COMM_WORLD);
                std::cout << "Controller sent abort signal to Producer " << i << std::endl;
            }
        }

        std::cout << "MPI Is Palindrome?: " << (isPalindrome == 0 ? "FALSE" : "TRUE") << std::endl;
    }
    else if (rank % 2 == 1) // Producer (odd rank) logic
    {
        std::cout << "Producer " << rank << " started" << std::endl;
        srand(time(nullptr) + rank); // different seed with the usage of rank

        MPI_Status status;
        int flag;
        MPI_Iprobe(controllerRank, abortSignalTag, MPI_COMM_WORLD, &flag, &status);
        if (flag)
        {
            int abortSignal;
            MPI_Recv(&abortSignal, 1, MPI_INT, controllerRank, abortSignalTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (abortSignal == 1)
            {
                abortSignalReceived = true;
                std::cout << "Producer " << rank << " received abort signal." << std::endl;
            }
        }

        // Check for abort signal from the controller
        // Artificial slowdown test here. (MS)
        // std::this_thread::sleep_for(std::chrono::milliseconds(rank * rank * rank));

        if (!abortSignalReceived)
        {

            std::vector<std::tuple<int, int, int>> tuples(numTuples);

            for (int i = 0; i < numTuples; i++)
            {
                int number = rand() % 4 + 1; // Generate a random number between 1 and 4
                tuples[i] = std::make_tuple(number, number, rank);

                // break a palindrome here by inserting 5

                // if (rank == 1 && i == 2)
                // {
                //     tuples[i] = std::make_tuple(number, 5, rank);
                // }
            }
            
            

            // Send the tuples to the "buddy" consumer (producer rank + 1)
            MPI_Send(tuples.data(), numTuples * 3, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            std::cout << "Producer " << rank << " sent " << numTuples << " tuples to consumer" << rank + 1 << std::endl;

            // // Send the tuples to the controller (rank 0) if we want to build the entire array to show (MS) - ADD RECEIVING LOGIC IN CONTROLLER
            // MPI_Send(tuples.data(), numTuples * 3, MPI_INT, controllerRank, 0, MPI_COMM_WORLD);
            // std::cout << "Producer " << rank << " sent " << numTuples << " tuples to controller" << std::endl;
        }
    }
    else if (rank % 2 == 0) // Consumer (even rank) logic
    {
        std::cout << "Consumer " << rank << " started" << std::endl;

        std::vector<int> rawReceivedData(numTuples * 3); // To store the flattened data

        // we can make a tuple out of raw received data if it is easier to work with it (MS)
        std::vector<std::tuple<int, int, int>> receivedTuples(numTuples);

        // Receive tuples from the "buddy" producer (rank - 1)
        int producerRank = rank - 1;
        MPI_Recv(rawReceivedData.data(), numTuples * 3, MPI_INT, producerRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Consumer " << rank << " received data from producer " << producerRank << ": " << std::endl;

        // Print the received data as tuples for debugging
        for (int i = 0; i < numTuples; ++i)
        {
            std::cout << "Tuple " << i + 1 << ": (" << rawReceivedData[i * 3] << ", " << rawReceivedData[i * 3 + 1] << ", " << rawReceivedData[i * 3 + 2] << ")" << std::endl;
        }

        // HEX output for debugging
        std::cout << std::hex << "Raw Data: ";
        for (const auto &val : rawReceivedData)
        {
            std::cout << val << " ";
        }
        std::cout << std::dec << std::endl;

        // TODO: rewrite to show an actual Producer rank, not an assumed one - (MS)
        std::cout << "Consumer " << rank << " received tuples from Producer " << rank - 1 << std::endl;

        int isPalindromeLocal = 1;
        // Consumer code that checks tuples received from its producer buddy
        for (int i = 0; i < numTuples; i++)
        {
            // std::cout << "Consumer" << rank << " is checking pair (" << rawReceivedData[i * 3] << "," << rawReceivedData[i * 3 + 1] << ") from Producer" << rawReceivedData[i * 3 + 2] << std::endl;
            // if (rawReceivedData[i * 3] != rawReceivedData[i * 3 + 1]) // If the pair is not the same (mac version)
            if (rawReceivedData[i * 3 + 1] != rawReceivedData[i * 3 + 2]) // If the pair is not the same (ubuntu version)
            {
                isPalindromeLocal = 0;
                std::cout << "Consumer" << rank << " found a mismatch and will send 0 to the controller." << std::endl;
            }
        }

        // If no mismatch is found
        MPI_Send(&isPalindromeLocal, 1, MPI_INT, controllerRank, 0, MPI_COMM_WORLD);
        std::cout << "Consumer " << rank << " found all pairs matching and sent 1 to the controller." << std::endl;
    }

    MPI_Finalize();
    return 0;
}

void printStatistics(const std::vector<int> &arr)
{
    if (arr.empty())
    {
        std::cout << "No data to display." << std::endl;
        return;
    }

    // Calculate the average
    double average = std::accumulate(arr.begin(), arr.end(), 0.0) / arr.size();

    // Calculate counts
    int countOnes = std::count(arr.begin(), arr.end(), 1);
    int countTwos = std::count(arr.begin(), arr.end(), 2);
    int countThrees = std::count(arr.begin(), arr.end(), 3);
    int countFours = std::count(arr.begin(), arr.end(), 4);
    int countZeros = std::count(arr.begin(), arr.end(), 0);
    int countRest = std::count_if(arr.begin(), arr.end(), [](int x)
                                  { return x > 4 || x < 0; });

    int countInRange = std::count_if(arr.begin(), arr.end(), [](int x)
                                     { return x >= 1 && x <= 4; });

    // Output statistics
    std::cout << "\nStatistics:\n";
    std::cout << "Average: " << std::fixed << std::setprecision(2) << average << "\n";
    std::cout << "Count (1-4): " << countInRange << "\n";

    std::cout << "\nNumber Distribution:\n";
    std::cout << "Ones: " << countOnes << "\n";
    std::cout << "Twos: " << countTwos << "\n";
    std::cout << "Threes: " << countThrees << "\n";
    std::cout << "Fours: " << countFours << "\n";
    std::cout << "Zeros: " << countZeros << "\n";
    std::cout << "Rest: " << countRest << "\n";
}
