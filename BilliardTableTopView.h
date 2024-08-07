/**
 * @file BilliardTableTopView.h
 * @author Laura Scarabello (laura.scarabello.1@studenti.unipd.it)
 * @brief # 2024175
 * @date 2024-07-11
 * 
 */

#include <opencv2/opencv.hpp>

class BilliardTableTopView {
public:
    
    void setTableCorners(const std::vector<cv::Point2f>& corners) {
        if (corners.size() != 4) {
            throw std::runtime_error("Sono richiesti esattamente 4 punti angolari.");
        }
        tableCorners = corners;
    }

    
    cv::Mat getTopView(const cv::Size& outputSize) {
    if (tableCorners.size() != 4) {
        throw std::runtime_error("Impostare prima i punti angolari del tavolo.");
    }

    
    std::vector<cv::Point2f> topViewCorners = {
        cv::Point2f(0, 0),
        cv::Point2f(static_cast<float>(outputSize.width), 0),
        cv::Point2f(static_cast<float>(outputSize.width), static_cast<float>(outputSize.height)),
        cv::Point2f(0, static_cast<float>(outputSize.height))
    };


   
    cv::Mat topViewImage = cv::Mat::zeros(outputSize, CV_8UC3);

    
    cv::Scalar tableColor(34, 139, 34); 

    
    topViewImage.setTo(tableColor);

    
    int holeRadius = 20;
    cv::circle(topViewImage, cv::Point(0, 0), holeRadius, cv::Scalar(0, 0, 0), -1);
    cv::circle(topViewImage, cv::Point(outputSize.width / 2, 0), holeRadius, cv::Scalar(0, 0, 0), -1);
    cv::circle(topViewImage, cv::Point(outputSize.width, 0), holeRadius, cv::Scalar(0, 0, 0), -1);
    cv::circle(topViewImage, cv::Point(0, outputSize.height), holeRadius, cv::Scalar(0, 0, 0), -1);
    cv::circle(topViewImage, cv::Point(outputSize.width / 2, outputSize.height), holeRadius, cv::Scalar(0, 0, 0), -1);
    cv::circle(topViewImage, cv::Point(outputSize.width, outputSize.height), holeRadius, cv::Scalar(0, 0, 0), -1);

    
    int lineThickness = 2;
    cv::line(topViewImage, cv::Point(0, 0), cv::Point(outputSize.width, 0), cv::Scalar(255, 255, 255), lineThickness);
    cv::line(topViewImage, cv::Point(outputSize.width, 0), cv::Point(outputSize.width, outputSize.height), cv::Scalar(255, 255, 255), lineThickness);
    cv::line(topViewImage, cv::Point(outputSize.width, outputSize.height), cv::Point(0, outputSize.height), cv::Scalar(255, 255, 255), lineThickness);
    cv::line(topViewImage, cv::Point(0, outputSize.height), cv::Point(0, 0), cv::Scalar(255, 255, 255), lineThickness);

    
    return topViewImage;
}

private:
    std::vector<cv::Point2f> tableCorners;
};
