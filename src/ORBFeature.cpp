#include "ORB/ORBFeature.hpp"

namespace ORB {
    ORBFeature::ORBFeature(const std::string& image_one_path, const std::string& image_two_path, const std::string& config_path)
    {
        camera_ptr = std::make_shared<Parameter>(config_path);
        image_one = cv::imread(image_one_path, cv::IMREAD_GRAYSCALE);
        image_two = cv::imread(image_two_path, cv::IMREAD_GRAYSCALE);

        orb_extractor_ptr = std::make_shared<ORBextractor>(camera_ptr->nFeatures, camera_ptr->scalseFactor, camera_ptr->nLevels, camera_ptr->iniThFAST, camera_ptr->minThFAST);
    }

    void ORBFeature::Run()
    {   // 作业1
        MatchImage();
        
        cv::Mat R, t;
        std::vector<cv::Point2f> vPoint_one, vPoint_two;
        // 作业2
        PoseEstimation(R, t, vPoint_one, vPoint_two);

        if (!R.empty())
        {
            std::cout << "R" << std::endl << R << std::endl;
            std::cout << "t" << std::endl << t << std::endl;
        }

        // 作业3
        std::vector<cv::Point3d> points;
        Triangulation(points);
        if (points.size() != 0)
        {
            std::cout << "三维点的深度值" << std::endl;
            for (int i = 0, id = points.size(); i != id; i++)
            {
                std::cout<< points[i].z << std::endl;
            }
        }
    }
    void ORBFeature::MatchImage()
    {
        std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
        cv::Mat descriptors_1, descriptors_2;
        std::vector<cv::DMatch> good_matches;

        // TODO
        FindFeatureMatches(image_one, image_two, keypoints_1, keypoints_2, good_matches);

        cv::Mat img_goodmatch;
        cv::drawMatches(image_one, keypoints_1, image_two, keypoints_2, good_matches, img_goodmatch);
        cv::imshow("good matches", img_goodmatch);
        cv::waitKey(0);
    }

    /**
     *  vPoint_one ： 图像1上的匹配特征点
     *  vPoint_two :  图像2上的匹配特征点
     *  TODO：调用了多次FindFeatureMatches函数
     */
    void ORBFeature::PoseEstimation(cv::Mat& R21, cv::Mat& t21, std::vector<cv::Point2f>& vPoint_one, std::vector<cv::Point2f>& vPoint_two)
    {
        /**************************TODO 根据前面的内容编程**********************************************/
        // 相机内参矩阵
        cv::Mat K = (cv::Mat_<double>(3, 3) << camera_ptr->fx, 0, camera_ptr->cx,
                                               0, camera_ptr->fy, camera_ptr->cy,
                                               0, 0, 1);
        
        // 在对极约束也需要匹配好的点对
        std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
        std::vector<cv::DMatch> good_matches;
        FindFeatureMatches(image_one, image_two, keypoints_1, keypoints_2, good_matches);

        // 把KeyPoint类型的特征点转为Point2f类型
        for (int i = 0; i< (int)good_matches.size(); i++)
        {
            vPoint_one.push_back(keypoints_1[good_matches[i].queryIdx].pt);
            vPoint_two.push_back(keypoints_2[good_matches[i].trainIdx].pt);
        }

        // 利用opencv的函数计算本质矩阵
        cv::Point2d principal_point(camera_ptr->cx, camera_ptr->cy);
        double focal_length = (camera_ptr->fx + camera_ptr->fy) / 2;
        cv::Mat essential_matrix;
        essential_matrix = cv::findEssentialMat(vPoint_one, vPoint_two, focal_length, principal_point);

        // 从本质矩阵中恢复R和t
        cv::recoverPose(essential_matrix, vPoint_one, vPoint_two, R21, t21, focal_length, principal_point);
    }

    /**
     *  points 是三维点的坐标
     */ 
    void ORBFeature::Triangulation(std::vector<cv::Point3d>& points)
    {
        /**************************TODO**********************************************/
        cv::Mat R,t;
        std::vector<cv::Point2f> pts1, pts2;
        PoseEstimation(R, t, pts1, pts2);

        cv::Mat K = (cv::Mat_<double>(3, 3) << camera_ptr->fx, 0, camera_ptr->cx,
                                               0, camera_ptr->fy, camera_ptr->cy,
                                               0, 0, 1);
        for (int i = 0; i < pts1.size(); i++) {
            pts1[i].x = (pts1[i].x - K.at<double>(0, 2)) / K.at<double>(0, 0);
            pts1[i].y = (pts1[i].y - K.at<double>(1, 2)) / K.at<double>(1, 1);
        }
        for (int i = 0; i < pts2.size(); i++) {
            pts2[i].x = (pts2[i].x - K.at<double>(0, 2)) / K.at<double>(0, 0);
            pts2[i].y = (pts2[i].y - K.at<double>(1, 2)) / K.at<double>(1, 1);
        }

        // image1的投影矩阵只有1
        cv::Mat T1 = (cv::Mat_<float>(3, 4)
        << 1, 0, 0, 0,
           0, 1, 0, 0,
           0, 0, 1, 0);
        
        // image2的投影矩阵
        cv::Mat T2 = (cv::Mat_<float>(3, 4)
        << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
           R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
           R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));
        
        

        cv::Mat pts_4d;
        cv::triangulatePoints(T1, T2, pts1, pts2, pts_4d);

        // 转换为非齐次坐标
        for (int i = 0; i < pts_4d.cols; i++)
        {
            cv::Mat x = pts_4d.col(i);
            x /= x.at<float>(3, 0); // 归一化
            cv::Point3d p(
                x.at<float>(0, 0),
                x.at<float>(1, 0),
                x.at<float>(2, 0)
            );
            points.push_back(p);
        }
    }

    void ORBFeature::FindFeatureMatches(const cv::Mat& src_image_one, const cv::Mat& src_image_two,
                                        std::vector<cv::KeyPoint>& vkeypoints_one, 
                                        std::vector<cv::KeyPoint>& vkeypoints_two,
                                        std::vector<cv::DMatch>& good_matches)
    {
        cv::Mat descriptors_one, descriptors_two;
        std::vector<cv::DMatch> matches;
        if (1)
        {   
            /***************************TODO**********************************/
            ExtractORB(src_image_one, vkeypoints_one, descriptors_one);
            ExtractORB(src_image_two, vkeypoints_two, descriptors_two);
        }
        else {
            ORBSLAM2ExtractORB(src_image_one, vkeypoints_one, descriptors_one);
            ORBSLAM2ExtractORB(src_image_two, vkeypoints_two, descriptors_two);
        }
        /***************************TODO**********************************/
        // TODO 在选择描述子距离时，按照高博设置的经验值30作为下限！！！

        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
        matcher->match(descriptors_one, descriptors_two, matches);

        // 匹配的筛选，选用经验值30作为下限
        auto min_max = std::minmax_element(matches.begin(), matches.end(),
                                           [](const cv::DMatch &m1, const cv::DMatch &m2) { return m1.distance < m2.distance; });
        double min_dist = min_max.first->distance;
        double max_dist = min_max.first->distance;

        double threshold;
        // 当距离大于最小距离的2倍时，认为有误，30为经验值下限
        threshold = (2 * min_dist > 30) ? 2 * min_dist : 30;
        for (int i = 0; i < descriptors_one.rows; i++) {
            if (matches[i].distance <= threshold)
                good_matches.push_back(matches[i]);
        }

    }

    cv::Scalar ORBFeature::get_color(float depth)
    {
        float up_th = 50, low_th = 10, th_range = up_th - low_th;
        if (depth > up_th) depth = up_th;
        if (depth < low_th) depth = low_th;
        return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
    }

    void ORBFeature::ExtractORB(const cv::Mat& image, std::vector<cv::KeyPoint>& vkeypoints, cv::Mat& descriptors)
    {
        // TODO 使用ORB特征点提取器，也可以尝试其它的。
        // 这里使用opencv自带的函数
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();          // 特征提取
        cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();    // 描述子提取

        detector->detect(image, vkeypoints);
        descriptor->compute(image, vkeypoints, descriptors);
    }

    void ORBFeature::UndistortImage(const cv::Mat& image, cv::Mat& outImage)
    {
        int rows = image.rows;
        int cols = image.cols;
        cv::Mat Image = cv::Mat(rows, cols, CV_8UC1);

        for (int v = 0; v < rows; v++)
        {
            for (int u = 0; u < cols; u++)
            {
                double x = (u - camera_ptr->cx)/camera_ptr->fx;
                double y = (v - camera_ptr->cy)/camera_ptr->fy;

                double r = sqrt(x * x + y * y);
                double r2 = r * r;
                double r4 = r2 * r2;

                double x_dis = x * (1 + camera_ptr->k1 * r2 + camera_ptr->k2 * r4) + 2 * camera_ptr->p1 * x * y + camera_ptr->p2 * (r2 + 2 * x * x);
                double y_dis = y * (1 + camera_ptr->k1 * r2 + camera_ptr->k2 * r4) + camera_ptr->p1 * (r2 + 2 * y * y) + 2 * camera_ptr->p2 * x * y;

                double u_dis = camera_ptr->fx * x_dis + camera_ptr->cx;
                double v_dis = camera_ptr->fy * y_dis + camera_ptr->cy;

                if (u_dis >= 0 && v_dis >= 0 && u_dis < cols && v_dis < rows)
                {
                    Image.at<uchar>(v, u) = image.at<uchar>((int)v_dis, (int)u_dis);
                }
                else {
                    Image.at<uchar>(v, u) = 0;
                }
            }
            outImage = Image;
        }
    }

    void ORBFeature::ORBSLAM2ExtractORB(const cv::Mat& srcImage, std::vector<cv::KeyPoint>& vkeypoints, cv::Mat& descriptors)
    {
        orb_extractor_ptr->operator()(srcImage, cv::Mat(), vkeypoints, descriptors);       
    }
}