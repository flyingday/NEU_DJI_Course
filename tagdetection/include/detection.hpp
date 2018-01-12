#ifndef DETECTION_HPP
#define DETECTION_HPP

#include <opencv2/core.hpp>

#include <vector>

namespace TagDetection {

class Detector {
    public:
        using PointList     = std::vector<cv::Point>;
        using PointMatrix   = std::vector<PointList>;
        using ContourList   = std::vector<PointList>;
        using HierarchyList = std::vector<cv::Vec4i>;
        using VertexList    = std::vector<int>;
    private:
        cv::Mat   m_img;
        PointList m_trace = {};
    public:
        bool drawQRCode(const cv::Mat &img);

        const cv::Mat& getImage() const noexcept;
    private:
        /**
         * Find all potential vertices with pre-calculated contour and hierarchy
         */
        VertexList findVertices(const ContourList &contour, const HierarchyList &hierarchy);

        /**
         * If vertices are more than 3, then filter them by checking existence of timing pattern
         */
        VertexList filterVertices(const VertexList &vert, const ContourList &contour);

        bool checkLine(cv::Point a[], cv::Point b[]);

        bool isTimingPattern(cv::Point a, cv::Point b);
};

} // namespace TagDetection

#endif // DETECTION_HPP
