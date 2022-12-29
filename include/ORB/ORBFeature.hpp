#ifndef ORB_ORBFEATURE_HPP_
#define ORB_ORBFEATURE_HPP_

#include "ORB/parameter.hpp"
#include "ORB/ORBextractor.h"

#include <opencv2/opencv.hpp>


namespace ORB {
    class ORBFeature{
        public:
            ORBFeature(const std::string& image_one_path, const std::string& image_two_path, const std::string& config_path);

            void Run();
        private:
            
            void MatchImage();

            void PoseEstimation(cv::Mat& R21, cv::Mat& t21, std::vector<cv::Point2f>& vPoint_one, std::vector<cv::Point2f>& vPoint_two);

            void Triangulation(std::vector<cv::Point3d>& points);
        private:
            void FindFeatureMatches(const cv::Mat& image_one, const cv::Mat& image_two,
                                    std::vector<cv::KeyPoint>& vkeypoints_one, std::vector<cv::KeyPoint>& vkeypoints_two,
                                    std::vector<cv::DMatch>& matches);
            inline cv::Scalar get_color(float depth);
        private:
            //1、 提取ORB特征点,使用opencv函数实现
            void ExtractORB(const cv::Mat& image, std::vector<cv::KeyPoint>& vkeypoints, cv::Mat& descriptors);
            //2、 图像去畸变
            void UndistortImage(const cv::Mat& image, cv::Mat& outImage);
            // 3、 orb-slam2特征点提取器
            void ORBSLAM2ExtractORB(const cv::Mat& image, std::vector<cv::KeyPoint>& vkeypoints, cv::Mat& descriptors);

        private:
            std::shared_ptr<Parameter> camera_ptr;
            std::shared_ptr<ORBextractor> orb_extractor_ptr;
            cv::Mat image_one;
            cv::Mat image_two;
    };
}

#endif