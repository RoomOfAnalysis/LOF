#pragma once

#define _USE_MATH_DEFINES
#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

namespace LOF
{
    /**
     * Class-based implementation for outlier detection only
     */
    class LocalOutlierFactor
    {
    private:
        int n_neighbors_;
        float contamination_;
        float offset_;

        int n_samples_fit_ = 0;
        std::vector<float> negative_outlier_factor_;
        std::pair<cv::Mat, cv::Mat> neighbors_indices_and_dists_fit_x_;
        cv::Mat lrd_;

    public:
        explicit LocalOutlierFactor(int n_neighbors = 20, float contamination = 0.0)
            : n_neighbors_(n_neighbors), contamination_(contamination), offset_(-1.5)
        {
            assert(n_neighbors > 0);
            assert(contamination >= 0.0 && contamination < 1.0);
        }

        /**
         * Fit the model to the training dataset
         */
        void fit(const cv::Mat& X)
        {
            n_samples_fit_ = X.rows;

            assert(n_samples_fit_ > 0);

            // Adjust n_neighbors if it's greater than the number of samples
            int effective_n_neighbors = std::max(1, std::min(n_neighbors_, n_samples_fit_ - 1));

            // Precompute all k-nearest neighbors using FLANN
            neighbors_indices_and_dists_fit_x_ = find_all_k_nearest_neighbors_flann(X, effective_n_neighbors);

            auto const& [indices, dists] = neighbors_indices_and_dists_fit_x_;

            // Calculate local reachability density for all points
            lrd_ = local_reachability_density(indices, dists, n_neighbors_);

            cv::Mat lrd_ratios_array = cv::Mat::zeros(dists.rows, dists.cols, dists.type());
            for (auto r = 0; r < dists.rows; ++r)
                for (auto c = 0; c < dists.cols; ++c)
                    lrd_ratios_array.at<float>(r, c) = lrd_.at<float>(indices.at<int>(r, c), 0) / lrd_.at<float>(r, 0);

            // Calculate LOF scores
            cv::Mat negative_outlier_factor_mat;
            cv::reduce(lrd_ratios_array, negative_outlier_factor_mat, 1, cv::REDUCE_AVG, lrd_.type());
            negative_outlier_factor_mat *= -1.0;

            negative_outlier_factor_mat.col(0).copyTo(negative_outlier_factor_);

            // Determine offset based on contamination
            if (contamination_ == 0.0)
            {                   // Auto contamination
                offset_ = -1.5; // Default inlier score around -1
            }
            else
            {
                // Sort negative outlier factors and set offset based on contamination
                auto sorted_factors = negative_outlier_factor_;
                std::sort(sorted_factors.begin(), sorted_factors.end());

                float index = contamination_ * (n_samples_fit_ - 1);
                int lower_idx = static_cast<int>(std::floor(index));
                int upper_idx = static_cast<int>(std::ceil(index));
                if (lower_idx == upper_idx)
                    offset_ = sorted_factors[lower_idx];
                else
                {
                    float weight = index - lower_idx;
                    offset_ = sorted_factors[lower_idx] * (1.0f - weight) + sorted_factors[upper_idx] * weight;
                }
            }
        }

        /**
         * Predict outlier labels for the fitted data
         */
        std::vector<int> fit_predict()
        {
            std::vector<int> predictions(n_samples_fit_);
            for (int i = 0; i < n_samples_fit_; ++i)
                predictions[i] = (negative_outlier_factor_[i] < offset_) ? -1 : 1;
            return predictions;
        }

        /**
         * Get the negative outlier factor values
         */
        const std::vector<float>& get_negative_outlier_factor() const { return negative_outlier_factor_; }

        /**
         * Get the offset value
         */
        float get_offset() const { return offset_; }

    private:
        /**
     * Precompute all k-nearest neighbors using FLANN
     */
        inline static std::pair<cv::Mat, cv::Mat> find_all_k_nearest_neighbors_flann(const cv::Mat& dataset, int k)
        {
            int n_points = dataset.rows;

            // Create FLANN index
            cv::flann::Index flann_index(dataset, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN);

            // Prepare query matrix (entire dataset)
            cv::Mat indices, dists;
            int actual_k = std::min(k + 1, n_points);

            flann_index.knnSearch(dataset, indices, dists, actual_k, cv::flann::SearchParams());
            indices = indices.colRange(1, actual_k);
            dists = dists.colRange(1, actual_k);

            cv::sqrt(dists, dists);

            return std::make_pair(indices, dists);
        }

        /**
     * Calculate local reachability density
     */
        inline static cv::Mat local_reachability_density(cv::Mat const& indices, cv::Mat const& dists, int k)
        {
            cv::Mat dist_k = cv::Mat::zeros(indices.rows, indices.cols, dists.type());
            for (auto r = 0; r < indices.rows; ++r)
                for (auto c = 0; c < indices.cols; ++c)
                    dist_k.at<float>(r, c) = dists.at<float>(indices.at<int>(r, c), k - 1);
            cv::max(dists, dist_k, dist_k);
            cv::Mat mean;
            cv::reduce(dist_k, mean, 1, cv::REDUCE_AVG, dist_k.type());
            return 1.0 / (mean + 1e-10);
        }
    };
} // namespace LOF