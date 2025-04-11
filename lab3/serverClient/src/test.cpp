#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cerrno>
#include <iomanip>

bool is_close(double a, double b, double epsilon = 1e-5) {
    return std::abs(a - b) <= epsilon;
}

bool is_near_integer(double value, double epsilon = 1e-3) {
    return std::fabs(value - std::round(value)) <= epsilon;
}

void test_sqrt_file(const std::string& filename, double epsilon) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    std::string line;
    int total = 0;
    int errors = 0;

    while (std::getline(file, line)) {
        total++;
        size_t x_pos = line.find("x = ");
        size_t result_pos = line.find(" → result = ");

        if (x_pos == std::string::npos || result_pos == std::string::npos) {
            std::cerr << "format fucked up " << total << std::endl;
            errors++;
            continue;
        }

        try {
            std::string x_str = line.substr(x_pos + 4, result_pos - (x_pos + 4));
            double x = std::stod(x_str);
            std::string result_str = line.substr(result_pos + 14);
            double expected = std::stod(result_str);
            double actual = std::sqrt(x);

            if (!is_close(actual, expected, epsilon)) {
                std::cerr << "Error in " << total << ": waited " << std::fixed << expected 
                          << ", get: " << actual << std::endl;
                errors++;
            }
        } catch (const std::exception& e) {
            std::cerr << "Parsing fucked up " << total << ": " << e.what() << std::endl;
            errors++;
        }
    }

    std::cout << "[sqrt] file: " << filename 
              << "\n    checked: " << total 
              << "\n    errors: " << errors << "\n" << std::endl;
}


void test_sin_file(const std::string& filename, double epsilon) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "file fucked up: " << filename << std::endl;
        return;
    }

    std::string line;
    int total = 0;
    int errors = 0;

    while (std::getline(file, line)) {
        total++;
        size_t x_pos = line.find("x = ");
        size_t result_pos = line.find(" → result = ");

        if (x_pos == std::string::npos || result_pos == std::string::npos) {
            std::cerr << "format fucked up " << total << std::endl;
            errors++;
            continue;
        }

        try {
            std::string x_str = line.substr(x_pos + 4, result_pos - (x_pos + 4));
            double x = std::stod(x_str);
            std::string result_str = line.substr(result_pos + 14);
            double expected = std::stod(result_str);
            double actual = std::sin(x);

            if (!is_close(actual, expected, epsilon)) {
                std::cerr << "Error in " << total << ": waited " << std::fixed << expected 
                          << ", get " << actual << std::endl;
                errors++;
            }
        } catch (const std::exception& e) {
            std::cerr << "Parsing fucked up " << total << ": " << e.what() << std::endl;
            errors++;
        }
    }

    std::cout << "[sin] file: " << filename 
              << "\n    checked: " << total 
              << "\n    Errors: " << errors << "\n" << std::endl;
}

void test_pow_file(const std::string& filename, double epsilon) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "file fucked up: " << filename << std::endl;
        return;
    }

    std::string line;
    int total = 0;
    int errors = 0;

    while (std::getline(file, line)) {
        total++;
        size_t x_pos = line.find("x = ");
        size_t y_pos = line.find(", y = ");
        size_t result_pos = line.find(" → result = ");

        if (x_pos == std::string::npos || y_pos == std::string::npos || result_pos == std::string::npos) {
            std::cerr << "format fucked up " << total << std::endl;
            errors++;
            continue;
        }

        try {
            // Парсинг входных данных
            std::string x_str = line.substr(x_pos + 4, y_pos - (x_pos + 4));
            double x = std::stod(x_str);
            std::string y_str = line.substr(y_pos + 6, result_pos - (y_pos + 6));
            double y = std::stod(y_str);
            std::string result_str = line.substr(result_pos + 14);
            double expected = std::stod(result_str);

            double actual = std::pow(x, y);

            bool is_error = false;
            if (std::floor(expected) != std::floor(actual)) {
                if (!is_close(actual, expected, epsilon)) {
                    if (fabs(fabs(expected) - fabs(actual)) > 1) {
                        std::cerr << "Error in line " << total << ": Expected " << expected 
                                  << ", got " << actual << std::endl;
                        errors++;
                    }
                }
            }
        }

        catch (const std::exception& e) {
            std::cerr << "Parsing error in line " << total << ": " << e.what() << std::endl;
            errors++;
        }
    }

    std::cout << "[pow] file: " << filename 
              << "\n    checked: " << total 
              << "\n    Errors: " << errors << "\n" << std::endl;
}

int main() {
    double epsilon = 1e-5;

    test_sqrt_file("/home/n.abramov/parallelki/lab3/serverClient/output/sqrt_results.txt", epsilon);
    test_sin_file("/home/n.abramov/parallelki/lab3/serverClient/output/sin_results.txt", epsilon);
    test_pow_file("/home/n.abramov/parallelki/lab3/serverClient/output/pow_results.txt", epsilon);

    return 0;
}