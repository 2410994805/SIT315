/*
Task: M4.T1D Project
Project Title: Hybrid MPI + OpenMP Emergency Traffic Control Analyser

Purpose:
This program analyses real-time traffic demand and emergency incidents across city zones.
It uses MPI for distributed zone ownership and OpenMP for local parallel analytics.
The system ranks critical zones and recommends signal extension and diversion actions.

Compile Command:
mpic++ -std=c++17 -fopenmp -O2 -o m4_emergency_traffic m4_main.cpp

Run Command:
mpirun -np 4 ./m4_emergency_traffic demo_m4_traffic.csv demo_m4_incidents.csv 3
*/

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

    /* Stores one validated traffic demand observation for one intersection. */
    struct TrafficRow
    {
        std::string hour;
        std::string zoneId;
        std::string intersectionId;
        long long vehicles = 0;
    };

    /* Stores one validated incident or emergency event for one zone and hour. */
    struct IncidentRow
    {
        std::string hour;
        std::string zoneId;
        long long severity = 0;
        long long emergencyVehicles = 0;
    };

    /* Stores the aggregated operational state for one zone within one hour. */
    struct ZoneHourStats
    {
        long long vehicles = 0;
        long long incidentSeverity = 0;
        long long emergencyVehicles = 0;
        long long distinctIntersections = 0;
        long long congestionScore = 0;
        int recommendedGreenExtension = 0;
        int diversionLevel = 0;
    };

    /* Stores one ranked result row that is returned to the coordinator. */
    struct RankedZone
    {
        std::string hour;
        std::string zoneId;
        ZoneHourStats stats;
    };

    /* Stores runtime evidence for later evaluation and demonstration. */
    struct RunMetrics
    {
        double receiveSeconds = 0.0;
        double trafficComputeSeconds = 0.0;
        double incidentComputeSeconds = 0.0;
        double mergeSeconds = 0.0;
        double totalRuntimeSeconds = 0.0;
        long long trafficRows = 0;
        long long incidentRows = 0;
        long long zoneHourGroups = 0;
    };

    using StatsMap = std::unordered_map<std::string, ZoneHourStats>;
    using IntersectionMap = std::unordered_map<std::string, std::unordered_map<std::string, bool>>;

    /* Removes surrounding whitespace from a token. */
    std::string trim(const std::string &text)
    {
        std::size_t begin = 0;
        while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])))
        {
            ++begin;
        }

        std::size_t end = text.size();
        while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])))
        {
            --end;
        }

        return text.substr(begin, end - begin);
    }

    /* Creates a lowercase copy for lightweight header detection. */
    std::string toLower(std::string text)
    {
        std::transform(text.begin(), text.end(), text.begin(),
                       [](unsigned char ch)
                       { return static_cast<char>(std::tolower(ch)); });
        return text;
    }

    /* Combines hour and zone into one stable key for aggregation. */
    std::string makeKey(const std::string &hour, const std::string &zoneId)
    {
        return hour + "|" + zoneId;
    }

    /* Splits the stable key back into hour and zone. */
    void splitKey(const std::string &key, std::string &hour, std::string &zoneId)
    {
        const std::size_t pos = key.find('|');
        if (pos == std::string::npos)
        {
            hour.clear();
            zoneId.clear();
            return;
        }

        hour = key.substr(0, pos);
        zoneId = key.substr(pos + 1);
    }

    /* Uses stable hash partitioning so each zone always belongs to one worker. */
    int ownerRankForZone(const std::string &zoneId, int worldSize)
    {
        if (worldSize < 2)
        {
            throw std::runtime_error("At least two MPI processes are required.");
        }

        const std::size_t hashed = std::hash<std::string>{}(zoneId);
        return static_cast<int>(hashed % (worldSize - 1)) + 1;
    }

    /* Recommends a practical traffic-control response that is easy to explain in the demo. */
    void finalizeOperationalPolicy(ZoneHourStats &stats)
    {
        stats.congestionScore =
            stats.vehicles +
            (stats.incidentSeverity * 120) +
            (stats.emergencyVehicles * 180) +
            (stats.distinctIntersections * 20);

        int greenExtension = 10;
        greenExtension += static_cast<int>(std::min<long long>(30, stats.emergencyVehicles * 4));
        greenExtension += static_cast<int>(std::min<long long>(20, stats.incidentSeverity * 2));

        int diversionLevel = 1;
        if (stats.congestionScore >= 2500 || stats.emergencyVehicles >= 4 || stats.incidentSeverity >= 8)
        {
            diversionLevel = 3;
        }
        else if (stats.congestionScore >= 1200 || stats.emergencyVehicles >= 2 || stats.incidentSeverity >= 4)
        {
            diversionLevel = 2;
        }

        stats.recommendedGreenExtension = greenExtension;
        stats.diversionLevel = diversionLevel;
    }

    /* Converts a diversion level into an operational label. */
    std::string diversionLabel(int diversionLevel)
    {
        if (diversionLevel >= 3)
        {
            return "Immediate emergency diversion";
        }
        if (diversionLevel == 2)
        {
            return "Adaptive rerouting";
        }
        return "Standard signal optimization";
    }

    /* Parses one traffic CSV line. */
    bool parseTrafficLine(const std::string &line, TrafficRow &rowOut)
    {
        if (trim(line).empty())
        {
            return false;
        }

        std::stringstream row(line);
        std::string hour;
        std::string zoneId;
        std::string intersectionId;
        std::string vehicles;

        if (!std::getline(row, hour, ','))
            return false;
        if (!std::getline(row, zoneId, ','))
            return false;
        if (!std::getline(row, intersectionId, ','))
            return false;
        if (!std::getline(row, vehicles))
            return false;

        try
        {
            rowOut.hour = trim(hour);
            rowOut.zoneId = trim(zoneId);
            rowOut.intersectionId = trim(intersectionId);
            rowOut.vehicles = std::stoll(trim(vehicles));
        }
        catch (...)
        {
            return false;
        }

        return !rowOut.hour.empty() && !rowOut.zoneId.empty() && !rowOut.intersectionId.empty();
    }

    /* Parses one incident CSV line. */
    bool parseIncidentLine(const std::string &line, IncidentRow &rowOut)
    {
        if (trim(line).empty())
        {
            return false;
        }

        std::stringstream row(line);
        std::string hour;
        std::string zoneId;
        std::string severity;
        std::string emergencyVehicles;

        if (!std::getline(row, hour, ','))
            return false;
        if (!std::getline(row, zoneId, ','))
            return false;
        if (!std::getline(row, severity, ','))
            return false;
        if (!std::getline(row, emergencyVehicles))
            return false;

        try
        {
            rowOut.hour = trim(hour);
            rowOut.zoneId = trim(zoneId);
            rowOut.severity = std::stoll(trim(severity));
            rowOut.emergencyVehicles = std::stoll(trim(emergencyVehicles));
        }
        catch (...)
        {
            return false;
        }

        return !rowOut.hour.empty() && !rowOut.zoneId.empty();
    }

    /* Reads a CSV file and skips the first header row when detected. */
    std::vector<std::string> readCsvLines(const std::string &path, const std::string &expectedHeaderToken)
    {
        std::ifstream input(path);
        if (!input)
        {
            throw std::runtime_error("Failed to open file: " + path);
        }

        std::vector<std::string> lines;
        std::string line;
        bool checkedFirstMeaningfulLine = false;

        while (std::getline(input, line))
        {
            const std::string clean = trim(line);
            if (clean.empty())
            {
                continue;
            }

            if (!checkedFirstMeaningfulLine)
            {
                checkedFirstMeaningfulLine = true;
                if (toLower(clean).find(expectedHeaderToken) != std::string::npos)
                {
                    continue;
                }
            }

            lines.push_back(clean);
        }

        return lines;
    }

    /* Converts a vector of lines into one MPI-safe text payload. */
    std::string joinLines(const std::vector<std::string> &lines)
    {
        std::ostringstream out;
        for (const auto &line : lines)
        {
            out << line << '\n';
        }
        return out.str();
    }

    /* Restores the original vector of lines after MPI transfer. */
    std::vector<std::string> splitLines(const std::string &blob)
    {
        std::vector<std::string> lines;
        std::stringstream input(blob);
        std::string line;

        while (std::getline(input, line))
        {
            if (!trim(line).empty())
            {
                lines.push_back(line);
            }
        }

        return lines;
    }

    /* Sends a variable-length string using a two-message protocol. */
    void sendString(int targetRank, int baseTag, const std::string &payload)
    {
        const int size = static_cast<int>(payload.size());
        MPI_Send(&size, 1, MPI_INT, targetRank, baseTag, MPI_COMM_WORLD);

        if (size > 0)
        {
            MPI_Send(payload.data(), size, MPI_CHAR, targetRank, baseTag + 1, MPI_COMM_WORLD);
        }
    }

    /* Receives a variable-length string using the same protocol. */
    std::string receiveString(int sourceRank, int baseTag)
    {
        int size = 0;
        MPI_Recv(&size, 1, MPI_INT, sourceRank, baseTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::string payload(size, '\0');
        if (size > 0)
        {
            MPI_Recv(payload.data(), size, MPI_CHAR, sourceRank, baseTag + 1, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }

        return payload;
    }

    /* Merges one local map into another while preserving all accumulated values. */
    void mergeStatsInto(StatsMap &target, const StatsMap &source)
    {
        for (const auto &item : source)
        {
            ZoneHourStats &slot = target[item.first];
            slot.vehicles += item.second.vehicles;
            slot.incidentSeverity += item.second.incidentSeverity;
            slot.emergencyVehicles += item.second.emergencyVehicles;
            slot.distinctIntersections += item.second.distinctIntersections;
        }
    }

    /* Computes local analytics using OpenMP thread-local maps to avoid lock contention. */
    StatsMap computeLocalStatsOpenMP(const std::vector<std::string> &trafficLines,
                                     const std::vector<std::string> &incidentLines,
                                     RunMetrics &metrics)
    {
        const int threadCount = std::max(1, omp_get_max_threads());
        std::vector<StatsMap> trafficThreadMaps(threadCount);
        std::vector<IntersectionMap> intersectionThreadMaps(threadCount);
        std::vector<StatsMap> incidentThreadMaps(threadCount);

        /* Parallelize the traffic-demand aggregation. */
        const auto trafficStart = std::chrono::steady_clock::now();

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            StatsMap &localStats = trafficThreadMaps[tid];
            IntersectionMap &localIntersections = intersectionThreadMaps[tid];

#pragma omp for schedule(static)
            for (int i = 0; i < static_cast<int>(trafficLines.size()); ++i)
            {
                TrafficRow row;
                if (!parseTrafficLine(trafficLines[i], row))
                {
                    continue;
                }

                const std::string key = makeKey(row.hour, row.zoneId);
                localStats[key].vehicles += row.vehicles;
                localIntersections[key][row.intersectionId] = true;
            }
        }

        const auto trafficEnd = std::chrono::steady_clock::now();
        metrics.trafficComputeSeconds =
            std::chrono::duration<double>(trafficEnd - trafficStart).count();

        /* Parallelize the incident and emergency aggregation separately. */
        const auto incidentStart = std::chrono::steady_clock::now();

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            StatsMap &localStats = incidentThreadMaps[tid];

#pragma omp for schedule(dynamic, 4)
            for (int i = 0; i < static_cast<int>(incidentLines.size()); ++i)
            {
                IncidentRow row;
                if (!parseIncidentLine(incidentLines[i], row))
                {
                    continue;
                }

                ZoneHourStats &stats = localStats[makeKey(row.hour, row.zoneId)];
                stats.incidentSeverity += row.severity;
                stats.emergencyVehicles += row.emergencyVehicles;
            }
        }

        const auto incidentEnd = std::chrono::steady_clock::now();
        metrics.incidentComputeSeconds =
            std::chrono::duration<double>(incidentEnd - incidentStart).count();

        /* Merge thread-local maps into one worker-owned map. */
        const auto mergeStart = std::chrono::steady_clock::now();

        StatsMap merged;

        for (int tid = 0; tid < threadCount; ++tid)
        {
            mergeStatsInto(merged, trafficThreadMaps[tid]);
            mergeStatsInto(merged, incidentThreadMaps[tid]);

            for (const auto &item : intersectionThreadMaps[tid])
            {
                merged[item.first].distinctIntersections += static_cast<long long>(item.second.size());
            }
        }

        for (auto &item : merged)
        {
            finalizeOperationalPolicy(item.second);
        }

        const auto mergeEnd = std::chrono::steady_clock::now();
        metrics.mergeSeconds = std::chrono::duration<double>(mergeEnd - mergeStart).count();

        metrics.trafficRows = static_cast<long long>(trafficLines.size());
        metrics.incidentRows = static_cast<long long>(incidentLines.size());
        metrics.zoneHourGroups = static_cast<long long>(merged.size());

        return merged;
    }

    /* Serializes worker-owned analytics into a compact text result. */
    std::string serializeStatsMap(const StatsMap &statsMap)
    {
        std::ostringstream out;

        for (const auto &item : statsMap)
        {
            std::string hour;
            std::string zoneId;
            splitKey(item.first, hour, zoneId);

            if (hour.empty() || zoneId.empty())
            {
                continue;
            }

            const ZoneHourStats &stats = item.second;
            out << hour << ','
                << zoneId << ','
                << stats.vehicles << ','
                << stats.incidentSeverity << ','
                << stats.emergencyVehicles << ','
                << stats.distinctIntersections << ','
                << stats.congestionScore << ','
                << stats.recommendedGreenExtension << ','
                << stats.diversionLevel << '\n';
        }

        return out.str();
    }

    /* Deserializes worker results back into ranked zone objects. */
    std::vector<RankedZone> deserializeRankedZones(const std::string &blob)
    {
        std::vector<RankedZone> rows;
        std::stringstream input(blob);
        std::string line;

        while (std::getline(input, line))
        {
            if (trim(line).empty())
            {
                continue;
            }

            std::stringstream row(line);
            std::string hour;
            std::string zoneId;
            std::string vehicles;
            std::string severity;
            std::string emergencyVehicles;
            std::string intersections;
            std::string score;
            std::string greenExtension;
            std::string diversionLevel;

            if (!std::getline(row, hour, ','))
                continue;
            if (!std::getline(row, zoneId, ','))
                continue;
            if (!std::getline(row, vehicles, ','))
                continue;
            if (!std::getline(row, severity, ','))
                continue;
            if (!std::getline(row, emergencyVehicles, ','))
                continue;
            if (!std::getline(row, intersections, ','))
                continue;
            if (!std::getline(row, score, ','))
                continue;
            if (!std::getline(row, greenExtension, ','))
                continue;
            if (!std::getline(row, diversionLevel))
                continue;

            try
            {
                RankedZone item;
                item.hour = trim(hour);
                item.zoneId = trim(zoneId);
                item.stats.vehicles = std::stoll(trim(vehicles));
                item.stats.incidentSeverity = std::stoll(trim(severity));
                item.stats.emergencyVehicles = std::stoll(trim(emergencyVehicles));
                item.stats.distinctIntersections = std::stoll(trim(intersections));
                item.stats.congestionScore = std::stoll(trim(score));
                item.stats.recommendedGreenExtension = std::stoi(trim(greenExtension));
                item.stats.diversionLevel = std::stoi(trim(diversionLevel));
                rows.push_back(item);
            }
            catch (...)
            {
            }
        }

        return rows;
    }

    /* Prints the highest-priority zones with recommended traffic-control actions. */
    void printTopZonesPerHour(const std::vector<RankedZone> &rows, int topN)
    {
        std::unordered_map<std::string, std::vector<RankedZone>> groupedByHour;

        for (const auto &row : rows)
        {
            groupedByHour[row.hour].push_back(row);
        }

        std::vector<std::string> hours;
        hours.reserve(groupedByHour.size());
        for (const auto &item : groupedByHour)
        {
            hours.push_back(item.first);
        }
        std::sort(hours.begin(), hours.end());

        std::cout << "\nCritical emergency-response zones by hour\n";

        for (const auto &hour : hours)
        {
            auto &bucket = groupedByHour[hour];

            std::sort(bucket.begin(), bucket.end(),
                      [](const RankedZone &a, const RankedZone &b)
                      {
                          if (a.stats.congestionScore != b.stats.congestionScore)
                          {
                              return a.stats.congestionScore > b.stats.congestionScore;
                          }
                          if (a.stats.emergencyVehicles != b.stats.emergencyVehicles)
                          {
                              return a.stats.emergencyVehicles > b.stats.emergencyVehicles;
                          }
                          return a.zoneId < b.zoneId;
                      });

            std::cout << "\nHour: " << hour << "\n";
            for (int i = 0; i < static_cast<int>(bucket.size()) && i < topN; ++i)
            {
                const RankedZone &zone = bucket[i];
                std::cout << "  " << (i + 1) << ". "
                          << zone.zoneId
                          << " | score=" << zone.stats.congestionScore
                          << " | vehicles=" << zone.stats.vehicles
                          << " | severity=" << zone.stats.incidentSeverity
                          << " | emergency=" << zone.stats.emergencyVehicles
                          << " | green extension=" << zone.stats.recommendedGreenExtension << " s"
                          << " | action=" << diversionLabel(zone.stats.diversionLevel)
                          << "\n";
            }
        }
    }

    /* Prints concise runtime evidence for evaluation and oral defence. */
    void printMetrics(double coordinatorReadSeconds,
                      double maxReceiveSeconds,
                      double maxTrafficComputeSeconds,
                      double maxIncidentComputeSeconds,
                      double maxMergeSeconds,
                      double maxRuntimeSeconds,
                      long long totalTrafficRows,
                      long long totalIncidentRows,
                      long long totalZoneHourGroups,
                      int worldSize)
    {
        std::cout << "\nEvaluation metrics\n";
        std::cout << "Worker ranks: " << (worldSize - 1) << "\n";
        std::cout << "Traffic rows: " << totalTrafficRows << "\n";
        std::cout << "Incident rows: " << totalIncidentRows << "\n";
        std::cout << "Zone-hour groups: " << totalZoneHourGroups << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Coordinator read time: " << coordinatorReadSeconds << " s\n";
        std::cout << "Worker receive time (max): " << maxReceiveSeconds << " s\n";
        std::cout << "Traffic compute time (max): " << maxTrafficComputeSeconds << " s\n";
        std::cout << "Incident compute time (max): " << maxIncidentComputeSeconds << " s\n";
        std::cout << "Local merge time (max): " << maxMergeSeconds << " s\n";
        std::cout << "Total runtime (max): " << maxRuntimeSeconds << " s\n";
        std::cout << "OpenMP threads per worker: " << omp_get_max_threads() << "\n";
    }

    /* Rank 0 reads both datasets, partitions by zone, gathers worker analytics, and prints final results. */
    void runCoordinator(const std::string &trafficPath,
                        const std::string &incidentsPath,
                        int worldSize,
                        int topN)
    {
        if (worldSize < 2)
        {
            throw std::runtime_error("Run with at least 2 MPI processes.");
        }

        const auto totalStart = std::chrono::steady_clock::now();

        /* Read the source datasets on the coordinator only. */
        const auto readStart = std::chrono::steady_clock::now();
        const std::vector<std::string> trafficLines = readCsvLines(trafficPath, "hour");
        const std::vector<std::string> incidentLines = readCsvLines(incidentsPath, "hour");
        const auto readEnd = std::chrono::steady_clock::now();
        const double coordinatorReadSeconds = std::chrono::duration<double>(readEnd - readStart).count();

        /* Partition traffic and incident rows by zone ownership. */
        std::vector<std::vector<std::string>> trafficChunks(worldSize);
        std::vector<std::vector<std::string>> incidentChunks(worldSize);

        for (const auto &line : trafficLines)
        {
            TrafficRow row;
            if (!parseTrafficLine(line, row))
            {
                continue;
            }
            trafficChunks[ownerRankForZone(row.zoneId, worldSize)].push_back(line);
        }

        for (const auto &line : incidentLines)
        {
            IncidentRow row;
            if (!parseIncidentLine(line, row))
            {
                continue;
            }
            incidentChunks[ownerRankForZone(row.zoneId, worldSize)].push_back(line);
        }

        /* Send worker-owned partitions to each worker rank. */
        for (int rank = 1; rank < worldSize; ++rank)
        {
            sendString(rank, 100, joinLines(trafficChunks[rank]));
            sendString(rank, 200, joinLines(incidentChunks[rank]));
        }

        /* Gather final ranked rows and evaluation metrics from workers. */
        std::vector<RankedZone> globalRows;
        globalRows.reserve(1024);

        double maxReceiveSeconds = 0.0;
        double maxTrafficComputeSeconds = 0.0;
        double maxIncidentComputeSeconds = 0.0;
        double maxMergeSeconds = 0.0;
        double maxRuntimeSeconds = 0.0;

        const long long totalTrafficRows = static_cast<long long>(trafficLines.size());
        const long long totalIncidentRows = static_cast<long long>(incidentLines.size());
        long long totalZoneHourGroups = 0;

        for (int rank = 1; rank < worldSize; ++rank)
        {
            const std::vector<RankedZone> workerRows = deserializeRankedZones(receiveString(rank, 300));
            globalRows.insert(globalRows.end(), workerRows.begin(), workerRows.end());

            double packet[6] = {0, 0, 0, 0, 0, 0};
            MPI_Recv(packet, 6, MPI_DOUBLE, rank, 400, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            maxReceiveSeconds = std::max(maxReceiveSeconds, packet[0]);
            maxTrafficComputeSeconds = std::max(maxTrafficComputeSeconds, packet[1]);
            maxIncidentComputeSeconds = std::max(maxIncidentComputeSeconds, packet[2]);
            maxMergeSeconds = std::max(maxMergeSeconds, packet[3]);
            maxRuntimeSeconds = std::max(maxRuntimeSeconds, packet[4]);
            totalZoneHourGroups += static_cast<long long>(packet[5]);
        }

        const auto totalEnd = std::chrono::steady_clock::now();
        maxRuntimeSeconds = std::max(maxRuntimeSeconds,
                                     std::chrono::duration<double>(totalEnd - totalStart).count());

        /* Print ranked operational priorities and the evaluation evidence. */
        printTopZonesPerHour(globalRows, topN);
        printMetrics(coordinatorReadSeconds,
                     maxReceiveSeconds,
                     maxTrafficComputeSeconds,
                     maxIncidentComputeSeconds,
                     maxMergeSeconds,
                     maxRuntimeSeconds,
                     totalTrafficRows,
                     totalIncidentRows,
                     totalZoneHourGroups,
                     worldSize);
    }

    /* Each worker receives one zone partition, processes it locally, and returns concise analytics. */
    void runWorker()
    {
        const auto totalStart = std::chrono::steady_clock::now();

        /* Receive worker-owned traffic and incident data. */
        const auto receiveStart = std::chrono::steady_clock::now();
        const std::vector<std::string> trafficLines = splitLines(receiveString(0, 100));
        const std::vector<std::string> incidentLines = splitLines(receiveString(0, 200));
        const auto receiveEnd = std::chrono::steady_clock::now();

        RunMetrics metrics;
        metrics.receiveSeconds = std::chrono::duration<double>(receiveEnd - receiveStart).count();

        /* Run the hybrid MPI plus OpenMP analytics on the local partition. */
        const StatsMap statsMap = computeLocalStatsOpenMP(trafficLines, incidentLines, metrics);
        sendString(0, 300, serializeStatsMap(statsMap));

        /* Return compact runtime metrics for evaluation. */
        const auto totalEnd = std::chrono::steady_clock::now();
        metrics.totalRuntimeSeconds = std::chrono::duration<double>(totalEnd - totalStart).count();

        double packet[6] = {
            metrics.receiveSeconds,
            metrics.trafficComputeSeconds,
            metrics.incidentComputeSeconds,
            metrics.mergeSeconds,
            metrics.totalRuntimeSeconds,
            static_cast<double>(metrics.zoneHourGroups)};

        MPI_Send(packet, 6, MPI_DOUBLE, 0, 400, MPI_COMM_WORLD);
    }

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    /* Validate the command-line arguments before any processing begins. */
    if (argc < 3)
    {
        if (worldRank == 0)
        {
            std::cerr << "Usage: mpirun -np <P> ./m4_emergency_traffic <traffic_csv> <incidents_csv> [topN]\n";
        }
        MPI_Finalize();
        return 1;
    }

    int topN = 5;

    /* Accept an optional top-N parameter while keeping a safe default value. */
    if (argc >= 4)
    {
        try
        {
            topN = std::stoi(argv[3]);
        }
        catch (...)
        {
            if (worldRank == 0)
            {
                std::cerr << "Invalid topN supplied. Using default value 5.\n";
            }
            topN = 5;
        }
    }

    try
    {
        /* Execute the coordinator path on rank 0 and the worker path elsewhere. */
        if (worldRank == 0)
        {
            runCoordinator(argv[1], argv[2], worldSize, topN);
        }
        else
        {
            runWorker();
        }
    }
    catch (const std::exception &ex)
    {
        if (worldRank == 0)
        {
            std::cerr << "Fatal error: " << ex.what() << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}