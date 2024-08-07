/**
 * @file BilliardAnalyzer.cpp
 * @author Laura Scarabello (laura.scarabello.1@studenti.unipd.it)
 * @brief # 2024175
 * @date 2024-06-25
 * 
 */

#include "BilliardAnalyzer.h"
#include <iostream>


BilliardAnalyzer::BilliardAnalyzer(const cv::Mat& inputImage) : image(inputImage.clone()) {
    if (image.empty()) {
        std::cerr << "Error loading image." << std::endl;
    }
}

BilliardAnalyzer::~BilliardAnalyzer() {
}

cv::Mat BilliardAnalyzer::preprocessImageTable(const cv::Mat& image, int erosionCount, int dilationCount) {
    // Apply thresholding
    cv::Mat thresholdedImage;

    cv::inRange(image, cv::Scalar(80,90,0), cv::Scalar(255,255,75), thresholdedImage);

    // Create structuring element for erosion
    cv::Mat erosionElement = cv::getStructuringElement(cv::MORPH_RECT,
                                                       cv::Size(3, 3)
                                                       );

    // Apply erosion
    cv::Mat erodedImage = thresholdedImage;
    for(size_t i = 0; i < erosionCount; i++) {

        cv::erode(erodedImage, erodedImage, erosionElement);

    }

    // Create structuring element for dilation
    cv::Mat dilationElement = cv::getStructuringElement(cv::MORPH_RECT,
                                                        cv::Size(3, 3)
                                                        );

    // Apply dilation
    cv::Mat outputImage = erodedImage;
    for(size_t i = 0; i < dilationCount; i++) {

        cv::dilate(outputImage, outputImage, dilationElement);

    }

    for(size_t i = 0; i < dilationCount*2; i++) {

        cv::dilate(outputImage, outputImage, dilationElement);

    }

    for(size_t i = 0; i < erosionCount*2; i++) {

        cv::erode(outputImage, outputImage, erosionElement);

    }

    return outputImage;
}

cv::Point2f BilliardAnalyzer::getIntersection(cv::Vec2f line1, cv::Vec2f line2) {
    float rho1 = line1[0], theta1 = line1[1];
    float rho2 = line2[0], theta2 = line2[1];

    double a1 = cos(theta1), b1 = sin(theta1);
    double a2 = cos(theta2), b2 = sin(theta2);

    cv::Mat A = (cv::Mat_<double>(2, 2) << a1, b1, a2, b2);
    cv::Mat B = (cv::Mat_<double>(2, 1) << rho1, rho2);

    cv::Mat X;
    cv::solve(A, B, X);

    return cv::Point2f(X.at<double>(0, 0), X.at<double>(1, 0));
}

std::vector<cv::Point2f> BilliardAnalyzer::detectTableEdges() {
    cv::Mat thres_img = preprocessImageTable(image);

    cv::Mat edges;
    cv::Canny(thres_img, edges, 100, 150);
    
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(edges, lines, 1, CV_PI / 180, 70);

    // Compute average theta
    float avgTheta = 0.f;
    for (size_t i = 0; i < lines.size(); ++i) {
        avgTheta += lines[i][1];
    }
    avgTheta /= lines.size();
    float rot = 0.f;
    if (avgTheta < 1.4f) {
        rot = CV_PI / 12;
    }

    std::vector<cv::Vec2f> horizontalLines;
    std::vector<cv::Vec2f> verticalLines;

    for (size_t i = 0; i < lines.size(); ++i) {
        float rho = lines[i][0], theta = lines[i][1];
        
        // Reverse line direction in order to have positive rho
        if (rho < 0) {
            lines[i][0] = -rho;
            lines[i][1] = theta - CV_PI;
        }

        auto line = lines[i];

        if (theta < ((CV_PI / 4) + rot) || theta > 3 * ((CV_PI / 4) + rot)) {
            verticalLines.push_back(line);
        } else {
            horizontalLines.push_back(line);
        }
    }    

    auto filterAndSelectLines = [](std::vector<cv::Vec2f>& lines) {
        std::sort(lines.begin(), lines.end(), [](const cv::Vec2f& a, const cv::Vec2f& b) {
                return a[0] < b[0];

        });

        std::vector<cv::Vec2f> selectedLines;
        std::vector<cv::Vec2f> ClusterLines;
        float maxDistance = 75;

        for (size_t i = 0; i < lines.size(); i++) {
            ClusterLines.push_back(lines[i]);
            for (size_t j = i + 1; j < lines.size(); j++) {
                float distance = std::abs(lines[j][0] - lines[i][0]);
                
                if ((distance < maxDistance) && j<(lines.size()-1)) { 
                    ClusterLines.push_back(lines[j]);
                }
                else{
                    float theta_avg = 0.f;
                    float rho_avg = 0.f;
                    for (size_t k = 0; k < ClusterLines.size(); k++){
                        rho_avg += ClusterLines[k][0];
                        theta_avg += ClusterLines[k][1];
                    }

                    rho_avg /= ClusterLines.size();
                    theta_avg /= ClusterLines.size();
                
                    cv::Vec2f avg_line;
                    avg_line[0] = rho_avg;
                    avg_line[1] = theta_avg;
                    
                    selectedLines.push_back(avg_line);

                    ClusterLines.clear();
                    i = j-1;
                    break;
                }
            }
        }
        return selectedLines;
    };

    horizontalLines = filterAndSelectLines(horizontalLines);
    verticalLines = filterAndSelectLines(verticalLines);

    std::vector<cv::Point2f> tablePoints;

    if (horizontalLines.size() == 2 && verticalLines.size() == 2) {
        cv::Point2f topLeft = getIntersection(horizontalLines[0], verticalLines[0]);
        tablePoints.push_back(topLeft);
        cv::Point2f topRight = getIntersection(horizontalLines[0], verticalLines[1]);
        tablePoints.push_back(topRight);
        cv::Point2f bottomLeft = getIntersection(horizontalLines[1], verticalLines[0]);
        tablePoints.push_back(bottomLeft);
        cv::Point2f bottomRight = getIntersection(horizontalLines[1], verticalLines[1]);
        tablePoints.push_back(bottomRight);
    } else {
        std::cerr << "It was not possible to find all the lines necessary to determine the edges of the table." << std::endl;
    }

    return tablePoints;
}

void BilliardAnalyzer::drawBoundingBox(std::vector<cv::Point2f> tablePoints, cv::Mat& image){

    cv::Point2f topLeft = tablePoints[0];
    cv::Point2f topRight = tablePoints[1];
    cv::Point2f bottomLeft = tablePoints[2];
    cv::Point2f bottomRight = tablePoints[3];

    cv::circle(image, topLeft, 5, cv::Scalar(0, 0, 255), -1);
    cv::circle(image, topRight, 5, cv::Scalar(0, 0, 255), -1);
    cv::circle(image, bottomLeft, 5, cv::Scalar(0, 0, 255), -1);
    cv::circle(image, bottomRight, 5, cv::Scalar(0, 0, 255), -1);

    cv::line(image, topLeft, topRight, cv::Scalar(0, 255, 0), 2);
    cv::line(image, topRight, bottomRight, cv::Scalar(0, 255, 0), 2);
    cv::line(image, bottomRight, bottomLeft, cv::Scalar(0, 255, 0), 2);
    cv::line(image, bottomLeft, topLeft, cv::Scalar(0, 255, 0), 2);
}

std::tuple<std::vector<cv::Point2f>, std::vector<float>> BilliardAnalyzer::locateBalls(cv::Mat& image, std::vector<cv::Point2f> tablePoints) {

    // Order the points in the correct order

    cv::Point2i topLeft = tablePoints[0];
    cv::Point2i topRight = tablePoints[1];
    cv::Point2i bottomRight = tablePoints[3];
    cv::Point2i bottomLeft = tablePoints[2];

    std::vector<cv::Point2i> tablePoints_i;

    tablePoints_i.clear();
    tablePoints_i.push_back(topLeft);
    tablePoints_i.push_back(topRight);
    tablePoints_i.push_back(bottomRight);
    tablePoints_i.push_back(bottomLeft);


    // Create mask for the table region and apply it to get only the inner region
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::fillConvexPoly(mask, tablePoints_i, cv::Scalar(255));
    cv::Mat tableRegion;
    image.copyTo(tableRegion, mask);

    // Create grayscale image
    cv::Mat hsv_img = tableRegion.clone();
    cv::cvtColor(hsv_img, hsv_img, cv::COLOR_BGR2HSV);

    // Split the channels
    std::vector<cv::Mat> channels;
    cv::split(hsv_img, channels);

    // Apply houghcircles_alt on the value channel
    std::vector<cv::Vec3f> circles;    
    cv::HoughCircles(channels[2], circles, cv::HOUGH_GRADIENT_ALT, 4, 2, 100, 0.5, 4, 14);

    // Scan the table region and detect the balls blobs
    cv::Mat thres_img;
    // Threshold the value channel
    cv::inRange(tableRegion, cv::Scalar(80,90,0), cv::Scalar(255,255,75), thres_img);

    // Create structuring element for erosion and dilation
    cv::Mat struct_elem = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

    // Dilate the image then erode it to close the gaps
    size_t n_rep = 3;
    // First erosion
    for (size_t i = 0; i < n_rep; i++) {
        cv::erode(thres_img, thres_img, struct_elem);
    }
    // Then dilation twice
    for (size_t i = 0; i < n_rep*2; i++) {
        cv::dilate(thres_img, thres_img, struct_elem);
    }
    // Then erode to restore the original size
    for (size_t i = 0; i < n_rep; i++) {
        cv::erode(thres_img, thres_img, struct_elem);
    }

    // Apply blob detection
    cv::SimpleBlobDetector::Params params;
    params.filterByArea = true;
    params.minArea = 50;
    params.maxArea = 1000;
    params.filterByCircularity = true;
    params.minCircularity = 0.3;
    params.filterByConvexity = true;
    params.minConvexity = 0.3;

    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(thres_img, keypoints);

    // Scan all keypoints and remove those with radius < 5
    for (size_t i = 0; i < keypoints.size(); i++) {
        if (keypoints[i].size/2.f < 7.f) {
            keypoints.erase(keypoints.begin() + i);
            i--;
        }
    }

    // Draw the keypoints on the image
    cv::drawKeypoints(image, keypoints, image, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Process detected keypoints
    std::vector<cv::Point2f> ballCenters;
    std::vector<float> ballRadii;

    ballCenters.clear();
    ballRadii.clear();

    for (const auto& keypoint : keypoints) {
        cv::Point2f center = keypoint.pt;
        float radius = keypoint.size / 2.0f;
        // Store the center and radius
        ballCenters.push_back(center);
        ballRadii.push_back(radius);
    }

    return std::make_tuple(ballCenters, ballRadii);
}

void BilliardAnalyzer::classifyBalls(const cv::Mat& image, const std::vector<cv::Point2f>& ballCenters, const std::vector<float>& ballRadii,
                                    cv::Point2f& whitePt, float& whiteRadius, cv::Point2f& blackPt, float& blackRadius,
                                    std::vector<cv::Point2f>& stripedBalls, std::vector<float>& stripedRadii,
                                    std::vector<cv::Point2f>& solidBalls, std::vector<float>& solidRadii) {
    // Convert the image to HSV
    cv::Mat gray_img, hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    cv::cvtColor(image, gray_img, cv::COLOR_BGR2GRAY);

    // Clear the vectors
    stripedBalls.clear();
    solidBalls.clear();
    stripedRadii.clear();
    solidRadii.clear();

    float meanMax(0), meanMin(255);

    for (size_t i = 0; i < ballCenters.size(); ++i) {
        cv::Point2f center = ballCenters[i];
        float radius = ballRadii[i];

        // Define a circular mask for the ball
        cv::Mat mask = cv::Mat::zeros(gray_img.size(), CV_8UC1);
        cv::circle(mask, center, static_cast<int>(radius), cv::Scalar(255), -1);

        // Create the ROI using the mask
        cv::Mat ballROI, ballROI_color;
        gray_img.copyTo(ballROI, mask);
        image.copyTo(ballROI_color, mask);

        // Ensure the mask is properly centered and within the image bounds
        if (center.x - radius < 0 || center.y - radius < 0 || center.x + radius >= gray_img.cols || center.y + radius >= gray_img.rows) {
            continue; // Skip balls that are too close to the image borders
        }
            
        // Crop the ballROI to the ball size
        cv::Rect roi(center.x - radius, center.y - radius, 2 * radius, 2 * radius);
        mask = mask(roi);
        cv::Mat croppedBallROI = ballROI(roi);
        cv::Mat croppedBallROI_color = ballROI_color(roi);

        // Convert the cropped ballROI to hsv
        cv::Mat hsv_croppedBallROI;
        cv::cvtColor(croppedBallROI_color, hsv_croppedBallROI, cv::COLOR_BGR2HSV);

        // Split and show the channels
        std::vector<cv::Mat> channels;
        cv::split(hsv_croppedBallROI, channels);
        // Desaturate the cropped ballROI
        channels[1] = channels[1] * .8;  
        channels[2] = channels[2] * .8;
        // Merge the channels
        hsv_croppedBallROI = cv::Mat::zeros(hsv_croppedBallROI.size(), CV_8UC3);
        cv::merge(channels, hsv_croppedBallROI);
        cv::cvtColor(hsv_croppedBallROI, hsv_croppedBallROI, cv::COLOR_HSV2BGR);
        cv::cvtColor(hsv_croppedBallROI, hsv_croppedBallROI, cv::COLOR_BGR2GRAY);

        // Threshold the cropped ballROI
        cv::threshold(hsv_croppedBallROI, hsv_croppedBallROI, 160, 255, cv::THRESH_BINARY);

        // Count white pixels ratio
        cv::Mat whitePixels;
        cv::inRange(hsv_croppedBallROI, cv::Scalar(255), cv::Scalar(255), whitePixels);
        float whiteRatio = cv::countNonZero(whitePixels) / (float)cv::countNonZero(mask);

        // Get mean and std of the cropped ballROI
        cv::Scalar mean, stddev;
        cv::meanStdDev(croppedBallROI, mean, stddev, mask);

        if (mean[0] > meanMax) {
            meanMax = mean[0];
            whitePt = center;
            whiteRadius = radius;
        } else if (mean[0] < meanMin) {
            meanMin = mean[0];
            blackPt = center;
            blackRadius = radius;
        } 

        // Get the single highest peak

        int peaks = 1;

        if (whiteRatio > 0.09f) {
            stripedBalls.push_back(center);
            stripedRadii.push_back(radius);
        } else {
            solidBalls.push_back(center);
            solidRadii.push_back(radius);
        }
        
    }

    // Text to display white and black ball
    std::string whiteText = "Cue ball";
    std::string blackText = "8 Ball";
    std::string stripedText = "Striped ball";
    std::string solidText = "Solid ball";

    // Write on the image
    cv::putText(image, whiteText, cv::Point(whitePt.x, whitePt.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, blackText, cv::Point(blackPt.x, blackPt.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    
    // Write the text on the image
    for (size_t i = 0; i < stripedBalls.size(); i++) {
        if (stripedBalls[i] == whitePt || stripedBalls[i] == blackPt) {
            // Remove the current ball from the vector
            stripedBalls.erase(stripedBalls.begin() + i);
            i--;
        }
        else {
            cv::putText(image, stripedText, cv::Point(stripedBalls[i].x, stripedBalls[i].y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        }
    }

    for (size_t i = 0; i < solidBalls.size(); i++) {
        if (solidBalls[i] == whitePt || solidBalls[i] == blackPt) {
            // Remove the current ball from the vector
            solidBalls.erase(solidBalls.begin() + i);
            i--;
        }
        else {
            cv::putText(image, solidText, cv::Point(solidBalls[i].x, solidBalls[i].y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        }
    }
}