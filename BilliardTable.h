/**
 * @file BilliardTable.h
 * @author Laura Scarabello (laura.scarabello.1@studenti.unipd.it)
 * @brief # 2024175
 * @date 2024-07-11
 * 
 */

#include <opencv2/opencv.hpp>
#include <vector>
#include <stdexcept>

enum BallType {
    CUE,
    EIGHT,
    STRIPED,
    SOLID
};

struct Ball {
    cv::Point2f center;
    float radius;
    BallType type;
};

class BilliardTable {
public:
    BilliardTable(const std::vector<cv::Point2f>& tablePoints, cv::Mat& image)
        : tablePoints(tablePoints), image(image) {
        if (tablePoints.size() != 4) {
            throw std::invalid_argument("You must provide exactly 4 points for the table.");
        }
    }

    void drawTable() {

        // Ensure the points are in the correct order to form a rectangle
        std::vector<cv::Point2f> orderedPoints = orderPoints(tablePoints);

        // Convert tablePoints from Point2f to Point for fillPoly
        std::vector<cv::Point> tablePointsInt;
        for (const auto& pt : orderedPoints) {
            tablePointsInt.push_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));
        }


        // Create a mask for filling
        cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
        std::vector<std::vector<cv::Point>> contours(1, tablePointsInt);
        cv::fillPoly(mask, contours, cv::Scalar(255));

        image.setTo(cv::Scalar(34, 139, 34), mask);

        // Draw the table edges
        cv::polylines(image, contours, true, cv::Scalar(0, 0, 0), 2);
    }

    void drawBalls(const std::vector<Ball>& balls) {
        for (const auto& ball : balls) {
            cv::Scalar color;
            switch (ball.type) {
                case CUE:
                    color = cv::Scalar(255, 255, 255); // White
                    break;
                case EIGHT:
                    color = cv::Scalar(0, 0, 0); // Black
                    break;
                case STRIPED:
                    color = cv::Scalar(0, 0, 255); // Red
                    break;
                case SOLID:
                    color = cv::Scalar(255, 0, 0); // Blue
                    break;
            }
            cv::circle(image, ball.center, cvRound(ball.radius), color, -1);
        }
    }

private:
    std::vector<cv::Point2f> tablePoints;
    cv::Mat& image;
    std::vector<cv::Point2f> orderPoints(const std::vector<cv::Point2f>& points) {
        std::vector<cv::Point2f> ordered(4);
        std::vector<float> sums, diffs;

        for (const auto& pt : points) {
            sums.push_back(pt.x + pt.y);
            diffs.push_back(pt.x - pt.y);
        }

        ordered[0] = points[std::min_element(sums.begin(), sums.end()) - sums.begin()]; // Top-left
        ordered[2] = points[std::max_element(sums.begin(), sums.end()) - sums.begin()]; // Bottom-right
        ordered[1] = points[std::min_element(diffs.begin(), diffs.end()) - diffs.begin()]; // Top-right
        ordered[3] = points[std::max_element(diffs.begin(), diffs.end()) - diffs.begin()]; // Bottom-left

        return ordered;
    }
};
