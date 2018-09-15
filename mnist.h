#ifndef MNIST_READER_HPP
#define MNIST_READER_HPP

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace mnist {
    inline uint32_t readHeader(const std::unique_ptr<char[]>& buffer, size_t position) {
        auto header = reinterpret_cast<uint32_t*>(buffer.get());
        auto value = *(header + position);
        return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
    }

    inline std::unique_ptr<char[]> readMnistFile(const std::string& path, uint32_t key) {
        std::ifstream file;
        file.open(path, std::ios::in | std::ios::binary | std::ios::ate);
        if (!file) {
            std::cout << "Error opening file" << std::endl;
            return {};
        }
        auto size = file.tellg();
        std::unique_ptr<char[]> buffer(new char[size]);
        file.seekg(0, std::ios::beg);
        file.read(buffer.get(), size);
        file.close();
        auto magic = readHeader(buffer, 0);
        if (magic != key) {
            std::cout << "Invalid magic number, probably not a MNIST file" << std::endl;
            return {};
        }
        auto count = readHeader(buffer, 1);
        if (magic == 0x803) {
            auto rows    = readHeader(buffer, 2);
            auto columns = readHeader(buffer, 3);
            if (size < count * rows * columns + 16) {
                std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
                return {};
            }
        } else if (magic == 0x801) {
            if (size < count + 8) {
                std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
                return {};
            }
        }
        return buffer;
    }

    struct dataset {
        std::vector< std::vector<uint8_t> > trainingImages;
        std::vector< std::vector<uint8_t> > testImages;
        std::vector<uint8_t> trainingLabels;
        std::vector<uint8_t> testLabels;
    };

    inline void readImageFile(std::vector< std::vector<uint8_t> >& images, const std::string& path, std::size_t limit) {
        auto buffer = readMnistFile(path, 0x803);
        if (buffer) {
            auto count   = readHeader(buffer, 1);
            auto rows    = readHeader(buffer, 2);
            auto columns = readHeader(buffer, 3);
            auto image_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 16);
            if (limit > 0 && count > limit) {
                count = limit;
            }
            images.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                images.emplace_back(std::vector<uint8_t>(1 * 28 * 28));
                for (size_t j = 0; j < rows * columns; ++j) {
                    auto pixel   = *image_buffer++;
                    images[i][j] = pixel;
                }
            }
        }
    }

    inline void readLabelFile(std::vector<uint8_t>& labels, const std::string& path, std::size_t limit = 0) {
        auto buffer = readMnistFile(path, 0x801);
        if (buffer) {
            auto count = readHeader(buffer, 1);
            auto label_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 8);
            if (limit > 0 && count > limit) {
                count = limit;
            }
            labels.resize(count);
            for (size_t i = 0; i < count; ++i) {
                auto label = *label_buffer++;
                labels[i]  = label;
            }
        }
    }

    inline std::vector< std::vector<uint8_t> >  readTrainingImages(const std::string& folder, std::size_t limit) {
        std::vector< std::vector<uint8_t> > images;
        readImageFile(images, folder + "/train-images-idx3-ubyte", limit);
        return images;
    }

    inline std::vector< std::vector<uint8_t> > readTestImages(const std::string& folder, std::size_t limit) {
        std::vector< std::vector<uint8_t> > images;
        readImageFile(images, folder + "/t10k-images-idx3-ubyte", limit);
        return images;
    }

    inline std::vector<uint8_t> readTrainingLabels(const std::string& folder, std::size_t limit) {
        std::vector<uint8_t> labels;
        readLabelFile(labels, folder + "/train-labels-idx1-ubyte", limit);
        return labels;
    }

    inline std::vector<uint8_t>readTestLabels(const std::string& folder, std::size_t limit) {
        std::vector<uint8_t> labels;
        readLabelFile(labels, folder + "/t10k-labels-idx1-ubyte", limit);
        return labels;
    }

    inline dataset readDataSetDirect(const std::string& folder, std::size_t training_limit = 0, std::size_t test_limit = 0) {
        dataset dataSet;
        dataSet.trainingImages = readTrainingImages(folder, training_limit);
        dataSet.trainingLabels = readTrainingLabels(folder, training_limit);
        dataSet.testImages = readTestImages(folder, test_limit);
        dataSet.testLabels = readTestLabels(folder, test_limit);
        return dataSet;
    }

    inline dataset readDataSet(const std::string& folder, std::size_t training_limit = 0, std::size_t test_limit = 0) {
        return readDataSetDirect(folder, training_limit, test_limit);
    }
}

#endif
