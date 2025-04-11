#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <functional>
#include <boost/program_options.hpp>

namespace opt = boost::program_options;

void initializeVector(std::vector<double>& vector, int n) {
    for (int j = 0; j < n; j++) {
        vector[j] = j;
    }
}

void initializeMatrix(std::vector<double>& matrix, int startRow, int endRow, int numCols) {
    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < numCols; j++) {
            matrix[i * numCols + j] = i + j;
        }
    }
}

void computeRows(std::vector<double>& matrix, std::vector<double>& vector, std::vector<double>& result, 
    int startRow, int endRow, int size) {
    for (int i = startRow; i < endRow; i++) {
        result[i] = 0;
        for (int j = 0; j < size; j++) {
            result[i] += matrix[i * size + j] * vector[j];
        }
    }
}


int main(int argc, char** argv) {
    int n = 40000;
    int m = 40000;
    int numThreads;
    try {
        opt::options_description desc("All options");
        desc.add_options()
            ("Threads", opt::value<int>()->default_value(1), "how many threads will be used")
            ("help", "show help message");

        opt::variables_map vm;
        opt::store(opt::parse_command_line(argc, argv, desc), vm);
        opt::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }

        numThreads = vm["Threads"].as<int>();
        if (numThreads <= 0) {
            std::cerr << "Error: Number of threads must be positive!" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    std::vector<double> matrix;
    std::vector<double> vector;
    std::vector<double> result;
    matrix.reserve(n * m);
    vector.reserve(m);
    result.reserve(n);

    std::vector<std::thread> threads;

    int threadid;
    int items_per_thread = m / numThreads;
    int low_bound;
    int up_bound;
    for (int i = 0; i < numThreads; ++i) {
        threadid = i;
        low_bound = threadid * items_per_thread;
        up_bound = (threadid == numThreads - 1) ? (m - 1) : (low_bound + items_per_thread - 1);
        threads.emplace_back(initializeMatrix, std::ref(matrix), low_bound, up_bound, m);
    }

    std::thread vectorThread(initializeVector, std::ref(vector), n);
    for (auto& t : threads) {
        t.join();
    }
    vectorThread.join();

    threads.clear();

    const auto start{std::chrono::steady_clock::now()};

    for (int i = 0; i < numThreads; ++i) {
        threadid = i;
        low_bound = threadid * items_per_thread;
        up_bound = (threadid == numThreads - 1) ? (m - 1) : (low_bound + items_per_thread - 1);
        threads.emplace_back(computeRows, std::ref(matrix), std::ref(vector), std::ref(result), low_bound, up_bound, m);
    }

    for (auto& t : threads) {
        t.join();
    }

    const auto end{std::chrono::steady_clock::now()};

    const std::chrono::duration<double> elapsed_seconds{end - start};


    std::cout << "Result vector: ";
    for (int i = 0; i < m; i++) { 
        std::cout << result[i] << " " << std::endl;
    }
    std::cout << std::endl;
    std::cout << "calculations: " << elapsed_seconds.count() << std::endl;

    return 0;
}
