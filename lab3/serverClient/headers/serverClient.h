#include <iostream>
#include <unordered_map>
#include <cmath>
#include <queue>
#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <functional>
#include <fstream>
#include <chrono>
#include <random>

template <typename T>
class Server {
public:
    Server() : running(false) {}

    void start() {
        if (!running) {
            running = true;
            server_thread = std::thread(&Server::process_tasks, this);
        }
    }

    void stop() {
        if (running) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                running = false;
                task_queue_condition.notify_all();
            }
            if (server_thread.joinable()) {
                server_thread.join();
            }
        }
    }

    size_t add_task(std::packaged_task<T()> task) {
        std::future<T> result_future = task.get_future();
        size_t id;
        {
            std::lock_guard<std::mutex> lock(mutex);
            id = next_id++;
            tasks.push(std::move(task));
            result_futures[id] = std::move(result_future);
        }
        task_queue_condition.notify_one();
        return id;
    }

    T request_result(size_t id_res) {
        std::future<T> future;
        {
            std::unique_lock<std::mutex> lock(mutex);
            auto it = result_futures.find(id_res);
            if (it == result_futures.end()) {
                throw std::runtime_error("we fucked up");
            }
            future = std::move(it->second);
            result_futures.erase(it);
        }
        return future.get();
    }

private:
    void process_tasks() {
        std::unique_lock<std::mutex> lock(mutex);
        while (running || !tasks.empty()) {

            task_queue_condition.wait(lock, [this] { 
                return !tasks.empty() || !running; 
            });

            while (!tasks.empty()) {
                auto task  = std::move(tasks.front());
                tasks.pop();
                lock.unlock();
                task();
                lock.lock();
            }
            if (!running) break;
        }
    }

    bool running;
    std::thread server_thread;
    std::mutex mutex;
    std::condition_variable task_queue_condition;
    std::queue<std::packaged_task<T()>> tasks;
    std::unordered_map<size_t, std::future<T>> result_futures;
    size_t next_id = 0;
};

template <typename T>
class Client {
public:
    Client(Server<T>& server, int num_tasks, const std::string& filename, std::function<T(T)> task_function)
        : server(server), num_tasks(num_tasks), filename(filename), task_function(task_function) {}

    Client(Server<T>& server, int num_tasks, const std::string& filename, std::function<T(T, T)> task_function2)
        : server(server), num_tasks(num_tasks), filename(filename), task_function2(task_function2) {}

    void run() {
        std::vector<size_t> task_ids;
        std::vector<T> args1;
        std::vector<T> args2;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(0, 5);
    
        for (int i = 0; i < num_tasks; ++i) {
            T input1 = dist(gen);
            T input2 = dist(gen);
    
            if (task_function2) {
                args1.push_back(input1);
                args2.push_back(input2);
                std::packaged_task<T()> task([=] { return task_function2(input1, input2); });
                size_t id = server.add_task(std::move(task));
                task_ids.push_back(id);
            } else {
                args1.push_back(input1);
                std::packaged_task<T()> task([=] { return task_function(input1); });
                size_t id = server.add_task(std::move(task));
                task_ids.push_back(id);
            }
        }
    
        std::ofstream file(filename);
        if (file.is_open()) {
            for (size_t i = 0; i < task_ids.size(); ++i) {
                T result = server.request_result(task_ids[i]);
                if (task_function2) {
                    file << "x = " << args1[i] << ", y = " << args2[i] << " → result = " << result << '\n';
                } else {
                    file << "x = " << args1[i] << " → result = " << result << '\n';
                }
            }
            file.close();
        } else {
            std::cerr << "file fucked up: " << filename << std::endl;
        }
    }

private:
    Server<T>& server;
    int num_tasks;
    std::string filename;
    std::function<T(T)> task_function;
    std::function<T(T, T)> task_function2;
};
