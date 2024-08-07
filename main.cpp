/**
 * @file main.cpp
 * @author Laura Scarabello (laura.scarabello.1@studenti.unipd.it)
 * @brief # 2024175
 * @date 2024-06-25
 * 
 */

#include "VideoProcessor.h"
#include "BilliardAnalyzer.h"
#include "BilliardTableTopView.h"
#include "BilliardTable.h"
#include <opencv2/highgui.hpp>
#include <iostream>


int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path>" << std::endl;
        return -1;
    }

    std::string videoPath = argv[1];
    VideoProcessor vp(videoPath);

    if (!vp.isOpened()) {
        std::cerr << "Failed to open video file." << std::endl;
        return -1;
    }

        cv::Mat frame;
        int count = 0;
        std::vector<cv::Point2f> tablePoints;
        std::vector<cv::Point2f> ballCenters;
        std::vector<float> ballRadii;

        while (true) {
            frame = vp.getNextFrame();

            if (frame.empty()) {
                break; 
            }

            cv::Mat boundingBoxFrame = frame.clone();
            BilliardAnalyzer analyzer(boundingBoxFrame);
            
            if (count == 0) {
                tablePoints = analyzer.detectTableEdges();
            }

            analyzer.drawBoundingBox(tablePoints, boundingBoxFrame);

            std::tie(ballCenters, ballRadii) = analyzer.locateBalls(boundingBoxFrame, tablePoints);

            cv::Point2f whitePt, blackPt;
            float whiteRadius, blackRadius;

            std::vector<cv::Point2f> stripedBalls, solidBalls;
            std::vector<float> stripedRadii, solidRadii;

            analyzer.classifyBalls(boundingBoxFrame, ballCenters, ballRadii, // input src parameters
                                    // output parameters
                                    whitePt, whiteRadius, blackPt, blackRadius,
                                    stripedBalls, stripedRadii,
                                    solidBalls, solidRadii);

            cv::imshow("Frame", boundingBoxFrame);
            cv::waitKey(0);

            cv::Mat segmentationFrame = frame.clone();
            BilliardTable table(tablePoints, segmentationFrame);

            table.drawTable();

           std::vector<Ball> balls = {
                { whitePt, whiteRadius, CUE },
                { blackPt, blackRadius, EIGHT }
            };

            for (size_t i = 0; i < stripedBalls.size(); i++) {
                balls.push_back({ stripedBalls[i], stripedRadii[i], STRIPED });
            }

            for (size_t i = 0; i < solidBalls.size(); i++) {
                balls.push_back({ solidBalls[i], solidRadii[i], SOLID });
            }

            table.drawBalls(balls);

            cv::imshow("Pool Table", segmentationFrame);
            cv::waitKey(0);

            BilliardTableTopView topViewTransformer;
            topViewTransformer.setTableCorners(tablePoints);

            float width = cv::norm(tablePoints[0] - tablePoints[1]);  
            float height = cv::norm(tablePoints[0] - tablePoints[2]); 

            cv::Size outputSize(static_cast<int>(width), static_cast<int>(height));


            cv::Mat topViewImage = topViewTransformer.getTopView(outputSize);
            cv::imshow("Top View Image", topViewImage);
            cv::waitKey(0);

            count++;
        }
    

    return 0;
}
