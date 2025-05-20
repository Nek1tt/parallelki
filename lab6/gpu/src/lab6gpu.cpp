#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>

namespace opt = boost::program_options;

// Вычисляем линейную интерполяцию между двумя точками (x1, y1) и (x2, y2) для значения x.
double linearInterpolation(double x, double x1, double y1, double x2, double y2) {
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1));
}

// Открываем файл для записи и сохраняем матрицу в указанный файл с заданной точностью форматирования
void saveMatrixToFile(const std::vector<double>& matrix, int size, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Can't open the file " << filename << " to writing." << std::endl;
        return;
    }

    int fieldWidth = 10;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            outputFile << std::setw(fieldWidth) << std::fixed << std::setprecision(4) << matrix[i * size + j];
        }
        outputFile << std::endl;
    }
    outputFile.close();
}

// Зануляем внутреннюю область матрицы
void zeroInternalArea(std::vector<double>& matrix, size_t size) {
    for (size_t i = 1; i < size - 1; ++i) {
        for (size_t j = 1; j < size - 1; ++j) {
            matrix[i * size + j] = 0.0;
        }
    }
}

// Инициализируем углы и края матрицы с использованием линейной интерполяции
void initializeMatrix(std::vector<double>& matrix, int size) {
    matrix[0] = 10.0;
    matrix[size - 1] = 20.0;
    matrix[(size - 1) * size + (size - 1)] = 30.0;
    matrix[(size - 1) * size] = 20.0;

    for (size_t i = 1; i < size - 1; i++) {
        matrix[0 * size + i] = linearInterpolation(i, 0.0, matrix[0], size - 1, matrix[size - 1]);
        matrix[i * size + 0] = linearInterpolation(i, 0.0, matrix[0], size - 1, matrix[(size - 1) * size]);
        matrix[i * size + (size - 1)] = linearInterpolation(i, 0.0, matrix[size - 1], size - 1, matrix[(size - 1) * size + (size - 1)]);
        matrix[(size - 1) * size + i] = linearInterpolation(i, 0.0, matrix[(size - 1) * size], size - 1, matrix[(size - 1) * size + (size - 1)]);
    }
}

int main(int argc, char const *argv[]) {
    opt::options_description desc("Опции");
    desc.add_options()
        ("accuracy", opt::value<double>()->default_value(1e-6), "Точность")
        ("size", opt::value<int>()->default_value(1024), "Размер матрицы")
        ("iterations", opt::value<int>()->default_value(1000000), "Количество итераций")
        ("help", "Помощь");

    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    int matrixSize = vm["size"].as<int>();
    double accuracy = vm["accuracy"].as<double>();
    int maxIterations = vm["iterations"].as<int>();

    double error = 1.0;
    int iteration = 0;

    std::vector<double> currentMatrix(matrixSize * matrixSize);
    std::vector<double> newMatrix(matrixSize * matrixSize);

    initializeMatrix(currentMatrix, matrixSize);
    initializeMatrix(newMatrix, matrixSize);
    zeroInternalArea(currentMatrix, matrixSize);
    zeroInternalArea(newMatrix, matrixSize);

    double* previousMatrix = newMatrix.data();
    double* updatedMatrix = currentMatrix.data();

    auto start = std::chrono::high_resolution_clock::now();
    #pragma acc data copyin(error, previousMatrix[0:matrixSize * matrixSize], updatedMatrix[0:matrixSize * matrixSize])
    {
        while (iteration < maxIterations && error > accuracy) {
            #pragma acc parallel loop present(updatedMatrix, previousMatrix)
            for (size_t i = 1; i < matrixSize - 1; i++) {
                for (size_t j = 1; j < matrixSize - 1; j++) {
                    updatedMatrix[i * matrixSize + j] = 0.25 * (
                        previousMatrix[i * matrixSize + j + 1] +
                        previousMatrix[i * matrixSize + j - 1] +
                        previousMatrix[(i - 1) * matrixSize + j] +
                        previousMatrix[(i + 1) * matrixSize + j]);
                }
            }
    
            if ((iteration + 1) % 10000 == 0) {
                error = 0.0;
                #pragma acc update device(error)
                #pragma acc parallel loop reduction(max:error) present(updatedMatrix, previousMatrix)
                for (size_t i = 1; i < matrixSize - 1; i++) {
                    for (size_t j = 1; j < matrixSize - 1; j++) {
                        error = std::max(error, std::abs(updatedMatrix[i * matrixSize + j] - previousMatrix[i * matrixSize + j]));
                    }
                }
                #pragma acc update self(error)
                std::cout << "Итерация: " << iteration + 1 << " Ошибка: " << error << std::endl;
            }
    
            std::swap(previousMatrix, updatedMatrix);
            iteration++;
        }
        #pragma acc update self(updatedMatrix[0:matrixSize * matrixSize])
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsedMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Time: " << elapsedMilliseconds << " ms, Error: " << error << ", Iterations: " << iteration << std::endl;

    if (matrixSize == 13 || matrixSize == 10) {
        for (size_t i = 0; i < matrixSize; i++) {
            for (size_t j = 0; j < matrixSize; j++) {
                std::cout << updatedMatrix[i * matrixSize + j] << ' ';
            }
            std::cout << std::endl;
        }
    }

    saveMatrixToFile(currentMatrix, matrixSize, "matrix.txt");
    return 0;
}