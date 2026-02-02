#pragma once

#include <opencv2/opencv.hpp>

#include <print>

namespace LOF::utils
{
    template <typename T> cv::Mat load_csv_to_mat(std::string const& filename)
    {
        std::ifstream file(filename);
        if (!file.is_open()) throw std::runtime_error("Error: Cannot open file " + filename);

        constexpr auto cv_t = std::is_same_v<T, double> ? CV_64F : CV_32F;

        std::string line;
        std::vector<std::vector<T>> data;
        size_t cols = 0;

        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string value;
            std::vector<T> row;

            while (std::getline(ss, value, ','))
            {
                try
                {
                    row.push_back(std::stod(value));
                }
                catch (const std::invalid_argument&)
                {
                    throw std::runtime_error("Error: Non-numeric value found in CSV.");
                }
            }

            if (cols == 0)
                cols = row.size();
            else if (row.size() != cols)
                throw std::runtime_error("Error: Inconsistent number of columns in CSV.");

            data.push_back(row);
        }

        // Create cv::Mat from the loaded data
        cv::Mat mat(static_cast<int>(data.size()), static_cast<int>(cols), cv_t);
        for (int i = 0; i < mat.rows; ++i)
            for (int j = 0; j < mat.cols; ++j)
                mat.at<T>(i, j) = data[i][j];
        return mat;
    }
} // namespace LOF::utils

template <> struct std::formatter<cv::Mat>: std::formatter<std::string>
{
    auto format(cv::Mat const& mat, format_context& ctx) const
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(8) << cv::format(mat, cv::Formatter::FMT_NUMPY);
        return std::formatter<std::string>::format(oss.str(), ctx);
    }
};