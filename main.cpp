#include <iostream>

#include "OpenVinoRunner.h"

enum Parameter {
    TF_MODEL_PATH = 1,
    ITERATION = 2,
    THREAD_COUNT = 3,
    TOTAL_COUNT = 4
};

int main(int argc, char*argv[]) {
    if (argc < TOTAL_COUNT) {
        std::cerr << "NO NO" << std::endl;
        return 1;
    }
    OpenVinoRunner runner;
    runner.init(argv[TF_MODEL_PATH], atoi(argv[THREAD_COUNT]));
    runner.run(atoi(argv[ITERATION]));

    return 0;
}