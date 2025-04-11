#include "serverClient.h"

template<typename T>
T fun_sin(T arg)
{
    return std::sin(arg);
}

template<typename T>
T fun_sqrt(T arg)
{
    return std::sqrt(arg);
}

template<typename T>
T fun_pow(T x, T y)
{
    return std::pow(x, y);
}

int main() {
    Server<double> server;
    server.start();

    Client<double> client1(server, 10000, "/home/n.abramov/parallelki/lab3/serverClient/output/sin_results.txt", fun_sin<double>);
    Client<double> client2(server, 10000, "/home/n.abramov/parallelki/lab3/serverClient/output/sqrt_results.txt", fun_sqrt<double>);
    Client<double> client3(server, 10000, "/home/n.abramov/parallelki/lab3/serverClient/output/pow_results.txt", fun_pow<double>);

    const auto start{std::chrono::steady_clock::now()};

    std::thread client_thread1(&Client<double>::run, &client1);
    std::thread client_thread2(&Client<double>::run, &client2);
    std::thread client_thread3(&Client<double>::run, &client3);

    client_thread1.join();
    client_thread2.join();
    client_thread3.join();

    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};

    server.stop();
    std::cout << "calculations: " << elapsed_seconds.count() << std::endl;
    return 0;
}