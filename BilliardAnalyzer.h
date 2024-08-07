/**
 * @file BilliardAnalyzer.h
 * @author Laura Scarabello (laura.scarabello.1@studenti.unipd.it)
 * @brief # 2024175
 * @date 2024-06-25
 * 
 */

#ifndef BILLIARDANALYZER_H
#define BILLIARDANALYZER_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

class BilliardAnalyzer {
public:
    BilliardAnalyzer(const cv::Mat& inputImage);
    ~BilliardAnalyzer();

    std::vector<cv::Point2f> detectTableEdges();
    void drawBoundingBox(std::vector<cv::Point2f> tablePoints, cv::Mat& image);
    std::tuple<std::vector<cv::Point2f>, std::vector<float>> locateBalls(cv::Mat& image, std::vector<cv::Point2f> tablePoints);
    void classifyBalls(const cv::Mat& image, const std::vector<cv::Point2f>& ballCenters, const std::vector<float>& ballRadii,
                                    cv::Point2f& whitePt, float& whiteRadius, cv::Point2f& blackPt, float& blackRadius,
                                    std::vector<cv::Point2f>& stripedBalls, std::vector<float>& stripedRadii,
                                    std::vector<cv::Point2f>& solidBalls, std::vector<float>& solidRadii);

    void setImage(const cv::Mat inputImage) {
        image = inputImage.clone();
    };

private:
    cv::Mat image;
    cv::Mat tableMask;
    std::vector<cv::Vec4i> tableEdges;
    std::vector<cv::Rect> ballPositions;

    cv::Mat preprocessImageTable(const cv::Mat& img, int erosionCount = 3, int dilationCount = 3);
    cv::Mat preprocessImageBalls(const cv::Mat& img, int erosionCount = 3, int dilationCount = 3);
    cv::Point2f getIntersection(cv::Vec2f line1, cv::Vec2f line2);
};

#endif // BILLIARDANALYZER_H
