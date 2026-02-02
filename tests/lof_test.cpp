#include "lof.hpp"
#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>

int main(int argc, char* argv[])
{
    using namespace LOF;
    using namespace LOF::utils;

    argparse::ArgumentParser program("chi2_shift_test");
    program.add_argument("--dataset").help("Path to dataset").required().metavar("FILE");
    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err)
    {
        std::cerr << err.what() << std::endl;
        std::exit(1);
    }

    auto dataset = load_csv_to_mat<float>(program.get<std::string>("--dataset"));

    std::println("dataset: {}", dataset);

    int k = dataset.cols;
    double contamination = 0.05;

    LocalOutlierFactor lof(std::min(k, 20), contamination);
    lof.fit(dataset);
    auto outliers = lof.fit_predict();
    std::println("outliers: {}", outliers);
    std::println("offset: {}", lof.get_offset());
    std::println("get_negative_outlier_factor: {}", lof.get_negative_outlier_factor());

    return 0;
}