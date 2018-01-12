#include "detection.hpp"

#include <cmath>
#include <utility>
#include <numeric>
#include <iostream>
#include <algorithm>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

#define ITERABLE(Container) Container.begin(), Container.end()

using PointList     = TagDetection::Detector::PointList;
using ContourList   = TagDetection::Detector::ContourList;
using HierarchyList = TagDetection::Detector::HierarchyList;
using VertexList    = TagDetection::Detector::VertexList;

static void testWindow(const Mat& img) {
    namedWindow(__func__);
    imshow(__func__, img);
    waitKey(0);
    destroyWindow(__func__);
}

template <class Callable>
static void testWindow(Mat& img, Callable &&call) {
    testWindow(call(img));
}

template <class T>
float variance(const vector<T> &arr) {
    float sum = accumulate(ITERABLE(arr), .0f);
    float mean = sum / static_cast<float>(arr.size());
    float squaresum = inner_product(ITERABLE(arr), arr.begin(), .0f);
    return squaresum / static_cast<float>(arr.size()) - mean*mean;
}

static inline pair<ContourList, HierarchyList> getContours(InputArray image) {
    Mat gray, ret;
    ContourList contours;
    HierarchyList hierarchy;
    // convert to grayscale
    cvtColor(image, gray, COLOR_BGR2GRAY);
    // apply canny edge detection
    GaussianBlur(gray, ret, Size{5,5}, 0);
    Canny(ret, gray, 100 , 200, 3);
    // Find contours with hierarchy
    findContours(gray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    return make_pair(move(contours), move(hierarchy));
}

static inline void minAreaBox(const PointList &points, Point dest[]) {
    Point2f result[4];
    auto rect = minAreaRect(points);
    rect.points(result);
    for(auto &p : result)
        *(dest++) = p;
}

template <class PtType = Point>
static inline PtType moment(InputArray pts, bool isBinaryImage = false) {
    auto m = moments(pts, isBinaryImage);
    return PtType(m.m10/m.m00, m.m01/m.m00);
}

static inline void adjustPoints(Point a[], Point b[]) {
    Point &a0 = a[0], &a1 = b[0], &b0 = a[1], &b1 = b[1];
    int dax = (a1.x-a0.x)/14, day = (a1.y-a0.y)/14,
        dbx = (b1.x-b0.x)/14, dby = (b1.y-b0.y)/14;
    a0.x += dax; a0.y += day; a1.x -= dax; a1.y -= day;
    b0.x += dbx; b0.y += dby; b1.x -= dbx; b1.y -= dby;
}

namespace TagDetection {

const Mat& Detector::getImage() const noexcept {
    return m_img;
}

bool Detector::drawQRCode(const Mat &img) {
    m_img = img.clone();

    ContourList contour;
    HierarchyList hierarchy;
    tie(contour, hierarchy) = getContours(m_img);
    auto vertices = findVertices(contour, hierarchy);

    if(vertices.size() != 3)
        return false;

    PointList pts, box{4};
    for(auto v : vertices)
        copy(ITERABLE(contour[v]), back_inserter(pts));
    minAreaBox(pts, box.data());

    polylines(m_img, ContourList{{box}}, true, Scalar{0,0,255}, 3);
    m_trace.emplace_back(moment(box));

//     ContourList outline;
//     for(auto v : vertices)
//         outline.emplace_back(move(contour[v]));
//     polylines(m_img, outline, true, Scalar{0,0,255}, 3);

    // draw trace
    for(auto &trace : m_trace)
        circle(m_img, trace, 1, Scalar{255, 0, 0}, 2);

    return true;
}

vector<int> Detector::findVertices(const ContourList &contour, const HierarchyList &hierarchy) {
    vector<int> vertices{};
    for(int i=0; i<contour.size(); ++i) {
        auto j = i, c = 0;
        while(hierarchy[j][2] != -1) {
            j = hierarchy[j][2];
            ++c;
        }
        if(c >= 5)
            vertices.push_back(i);
    }
    if(vertices.size() > 3)
        return filterVertices(vertices, contour);
    return vertices;
}

VertexList Detector::filterVertices(const VertexList &vertices, const ContourList &contour) {
    VertexList result;
    // iterate through lines between random vertices
    auto end = vertices.end();
    for(auto i=vertices.begin(); i != end; ++i) {
        for(auto j=i+1; j != end; ++j) {
            Point pts[8];
            minAreaBox(contour[*i], pts);
            minAreaBox(contour[*j], pts+4);
            if(checkLine(pts, pts+4)) {
                result.push_back(*i);
                result.push_back(*j);
            }
        }
    }

    // remove duplicates
    sort(ITERABLE(result));
    result.erase(unique(ITERABLE(result)), result.end());

    return result;
}

bool Detector::isTimingPattern(Point a, Point b) {
    Mat bin;
    threshold(m_img, bin, 100, 255, CV_THRESH_BINARY);
    LineIterator line{bin, a, b};
    vector<uchar> data(line.count);
    for(auto &p : data) {
        p = **line;
        ++line;
    }

    auto beg = data.begin(), end = data.end();
    while(*beg != 0) ++beg; // discard leading white pixels
    while(*(end - 1) != 0) --end; // discard trailing white pixels

    if(beg >= end)
        return false;

    vector<int> pixels; int count = 1;
    for(auto lbeg = beg + 1; lbeg != end; beg = lbeg++) {
        if(*lbeg == *beg)
            ++count;
        else {
            pixels.push_back(count);
            count = 1;
        }
    }

    return pixels.size() >= 5 &&
           variance(pixels) < 25; // this 25 is threshold
}

bool Detector::checkLine(Point a[], Point b[]) {
    static auto distance = [](Point a, Point b) {
        auto dx = a.x - b.x, dy = a.y - b.y;
        return dx*dx + dy*dy;
    };

    auto s1 = numeric_limits<int>::max(), s2 = s1;
    Point temp[4], *s1ab = temp, *s2ab = temp+2;
    for(int i=0; i<4; ++i) {
        for(int j=0; j<4; ++j) {
            auto &pa = a[i], &pb = b[j];
            auto d = distance(pa, pb);
            if(d < s2) {
                if(d < s1) {
                    swap(s2ab, s1ab);
                    s1ab[0] = pa; s1ab[1] = pb;
                    s2 = s1; s1 = d;
                } else {
                    s2ab[0] = pa; s2ab[1] = pb;
                    s2 = d;
                }
            }
        }
    }

    adjustPoints(s1ab, s2ab);
    return isTimingPattern(s1ab[0], s1ab[1]) ||
           isTimingPattern(s2ab[0], s2ab[1]);
}

} // namespace TagDetection
