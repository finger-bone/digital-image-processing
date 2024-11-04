#ifndef IMAGE_PROCESSING_NUMERIC_ARRAY_H
#define IMAGE_PROCESSING_NUMERIC_ARRAY_H

#include <vector>
#include <thread>
#include <functional>
#include <future>
#include <iostream>
#include <fstream>

namespace NumericArray
{
    template <typename T>
    struct NumericArray;

    template <typename T, typename U, typename V>
    NumericArray<T> binary_operation(const NumericArray<U> &a, const NumericArray<V> &b, std::function<T(U, V)> func,
                                     int workers = std::thread::hardware_concurrency())
    {
        int data_size = a.data.size();
        if (data_size != b.data.size())
        {
            throw std::runtime_error("Channels must have the same size");
        }
        if (data_size == 0)
            return NumericArray<T>();

        int chunk_size = data_size / workers;
        std::vector<std::future<std::vector<T>>> futures;

        for (int i = 0; i < workers; ++i)
        {
            int start_index = i * chunk_size;
            int end_index = (i == workers - 1) ? data_size : start_index + chunk_size;

            // Each thread processes a chunk of the data
            futures.emplace_back(std::async(std::launch::async, [&, start_index, end_index]()
                                            {
                std::vector<T> result(end_index - start_index);
                for (int j = start_index; j < end_index; ++j) {
                    result[j - start_index] = func(a.data[j], b.data[j]);
                }
                return result; }));
        }

        // Collect and combine the results from all threads
        std::vector<T> result;
        result.reserve(data_size);
        for (auto &future : futures)
        {
            auto result_chunk = future.get();
            result.insert(result.end(), result_chunk.begin(), result_chunk.end());
        }

        return NumericArray<T>{result};
    }

    template <typename T>
    struct NumericArray
    {
        std::vector<T> data;

        NumericArray() = default;

        explicit NumericArray(std::vector<T> data) : data(std::move(data)) {}

        explicit NumericArray(int size, T value) : data(size, value) {}

        std::vector<std::vector<T>> interpret(int height, int width, int workers = std::thread::hardware_concurrency()) {
            if (data.size() != height * width) {
                throw std::runtime_error("Data size does not match the expected size");
            }
            std::vector<std::vector<T>> result(height, vector<T>(width));
            this->foreach([&](T value, size_t index) {
                result[index / width][index % width] = value;
            }, workers);
            return result;
        }

        void foreach(std::function<void(T, size_t)> func, int workers = std::thread::hardware_concurrency())
        {
            int data_size = data.size();
            int chunk_size = data_size / workers;
            std::vector<std::future<void>> futures;
            for (int i = 0; i < workers; ++i)
            {
                int start_index = i * chunk_size;
                int end_index = (i == workers - 1) ? data_size : start_index + chunk_size;
                futures.emplace_back(std::async(std::launch::async, [this, func, start_index, end_index]()
                                                {
            for (int j = start_index; j < end_index; ++j) {
                func(data[j], j);
            } }));
            }
            for (auto &future : futures)
            {
                future.get();
            }
        }

        template <typename U>
        NumericArray<U> map(std::function<U(T, size_t)> func, int workers = std::thread::hardware_concurrency())
        {
            int data_size = data.size();
            if (data_size == 0)
                return NumericArray<U>();

            int chunk_size = data_size / workers;
            std::vector<std::future<std::vector<U>>> futures;

            for (int i = 0; i < workers; ++i)
            {
                int start_index = i * chunk_size;
                int end_index = (i == workers - 1) ? data_size : start_index + chunk_size;
                futures.emplace_back(std::async(std::launch::async, [this, func, start_index, end_index]()
                                                {
            std::vector<U> result;
            result.reserve(end_index - start_index);
            for (int j = start_index; j < end_index; ++j) {
                result.push_back(func(data[j], j));
            }
            return result; }));
            }

            std::vector<U> result;
            for (auto &future : futures)
            {
                auto result_chunk = future.get();
                result.insert(result.end(), result_chunk.begin(), result_chunk.end());
            }

            return NumericArray<U>{result};
        }

        void map_inplace(std::function<T(T, size_t)> func, int workers = std::thread::hardware_concurrency())
        {
            int data_size = data.size();
            if (data_size == 0)
                return;

            int chunk_size = data_size / workers;
            std::vector<std::future<void>> futures;

            for (int i = 0; i < workers; ++i)
            {
                int start_index = i * chunk_size;
                int end_index = (i == workers - 1) ? data_size : start_index + chunk_size;

                futures.emplace_back(std::async(std::launch::async, [this, func, start_index, end_index]()
                                                {
            for (int j = start_index; j < end_index; ++j) {
                data[j] = func(data[j], j);
            } }));
            }

            for (auto &future : futures)
            {
                future.get();
            }
        }

        void foreach (std::function<void(T)> func, int workers = std::thread::hardware_concurrency())
        {
            int data_size = data.size();
            int chunk_size = data_size / workers;
            std::vector<std::future<void>> futures;
            for (int i = 0; i < workers; ++i)
            {
                int start_index = i * chunk_size;
                int end_index = (i == workers - 1) ? data_size : start_index + chunk_size;
                futures.emplace_back(std::async(std::launch::async, [this, func, start_index, end_index]()
                                                {
                    for (int j = start_index;
                         j < end_index; ++j) {
                        func(data[j]);
                    } }));
            }
            for (auto &future : futures)
            {
                future.get();
            }
        }

        template <typename U>
        NumericArray<U> map(std::function<U(T)> func, int workers = std::thread::hardware_concurrency())
        {
            int data_size = data.size();
            if (data_size == 0)
                return NumericArray<U>();

            int chunk_size = data_size / workers;
            std::vector<std::future<std::vector<U>>> futures;

            for (int i = 0; i < workers; ++i)
            {
                int start_index = i * chunk_size;
                int end_index = (i == workers - 1) ? data_size : start_index + chunk_size;
                futures.emplace_back(std::async(std::launch::async, [this, func, start_index, end_index]()
                                                {
                    std::vector<U> result;
                    result.reserve(end_index - start_index);
                    for (int j = start_index; j < end_index; ++j) {
                        result.push_back(func(data[j]));
                    }
                    return result; }));
            }

            std::vector<U> result;
            for (auto &future : futures)
            {
                auto result_chunk = future.get();
                result.insert(result.end(), result_chunk.begin(), result_chunk.end());
            }

            return NumericArray<U>{result};
        }

        void map_inplace(std::function<T(T)> func, int workers = std::thread::hardware_concurrency())
        {
            int data_size = data.size();
            if (data_size == 0)
                return;

            int chunk_size = data_size / workers;
            std::vector<std::future<void>> futures;

            for (int i = 0; i < workers; ++i)
            {
                int start_index = i * chunk_size;
                int end_index = (i == workers - 1) ? data_size : start_index + chunk_size;

                futures.emplace_back(std::async(std::launch::async, [this, func, start_index, end_index]()
                                                {
                    for (int j = start_index; j < end_index; ++j) {
                        data[j] = func(data[j]);
                    } }));
            }

            for (auto &future : futures)
            {
                future.get();
            }
        }

        NumericArray<T> operator+(NumericArray<T> other)
        {
            return binary_operation(*this, other, [](T a, T b)
                                    { return a + b; });
        }

        NumericArray<T> operator-(NumericArray<T> other)
        {
            return binary_operation(*this, other, [](T a, T b)
                                    { return a - b; });
        }

        NumericArray<T> operator*(NumericArray<T> other)
        {
            return binary_operation(*this, other, [](T a, T b)
                                    { return a * b; });
        }

        NumericArray<T> operator/(NumericArray<T> other)
        {
            return binary_operation(*this, other, [](T a, T b)
                                    { return a / b; });
        }

        void set(T value)
        {
            data.assign(data.size(), value);
        }
    };
}

#endif // IMAGE_PROCESSING_NUMERIC_ARRAY_H
