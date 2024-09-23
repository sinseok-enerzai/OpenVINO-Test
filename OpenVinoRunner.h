#ifndef OPENVINO_RUNNER
#define OPENVINO_RUNNER

#include "openvino/openvino.hpp"

#include <chrono>
#include <vector>
#include <sstream>
#include <locale>
#include <algorithm>
#include <numeric>

constexpr const char *DEFALUT_DEVICE = "CPU";

void printInputAndOutputsInfo(const ov::Model &network)
{
    std::cout << "model name: " << network.get_friendly_name() << std::endl;

    const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
    for (const ov::Output<const ov::Node> &input : inputs)
    {
        std::cout << "    inputs" << std::endl;

        const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
        std::cout << "        input name: " << name << std::endl;

        const ov::element::Type type = input.get_element_type();
        std::cout << "        input type: " << type << std::endl;

        const ov::Shape shape = input.get_shape();
        std::cout << "        input shape: " << shape << std::endl;
    }

    const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
    for (const ov::Output<const ov::Node> &output : outputs)
    {
        std::cout << "    outputs" << std::endl;

        const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
        std::cout << "        output name: " << name << std::endl;

        const ov::element::Type type = output.get_element_type();
        std::cout << "        output type: " << type << std::endl;

        const ov::Shape shape = output.get_shape();
        std::cout << "        output shape: " << shape << std::endl;
    }
}

class OpenVinoRunner
{
public:
    bool init(const char *tfModelPath, unsigned int threadCount)
    {
        core.set_property(DEFALUT_DEVICE, ov::inference_num_threads(threadCount));
        // core.set_property(DEFALUT_DEVICE, ov::hint::enable_hyper_threading(true));
        core.set_property(DEFALUT_DEVICE, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));

        model = core.read_model(tfModelPath);
        printInputAndOutputsInfo(*model);
        return true;
    }

    void run(unsigned int iteration)
    {
        ov::CompiledModel compiledModel = core.compile_model(model, DEFALUT_DEVICE);
        std::cout << compiledModel.get_property(ov::inference_num_threads) << std::endl;
        ov::InferRequest infer_request = compiledModel.create_infer_request();
        times.reserve(iteration);

        for (unsigned int i = 0; i < iteration; ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            infer_request.start_async();
            infer_request.wait();
            auto end = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
        }

        print();
    }

private:
    void print()
    {
        int64_t sum = std::accumulate(times.begin(), times.end(), 0LL);
        std::sort(times.begin(), times.end());

        std::stringstream ss;
        std::cout.imbue(std::locale("en_US.UTF-8"));

        std::cout << "======================" << std::endl;
        std::cout << "average: " << sum / times.size() << "µs" << std::endl;
        std::cout << "median : " << times[times.size() / 2] << "µs" << std::endl;
        std::cout << "max    : " << times.back() << "µs" << std::endl;
        std::cout << "min    : " << times.front() << "µs" << std::endl;
    }

    ov::Core core;
    std::shared_ptr<ov::Model> model;
    std::vector<int64_t> times;
};

#endif // OPENVINO_RUNNER