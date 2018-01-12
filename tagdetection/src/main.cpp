#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <cassert>

#include "detection.hpp"

using namespace cv;
using namespace TagDetection;

int main() {
    VideoCapture cap{0};
    assert(cap.isOpened());

    Detector detector{};

    namedWindow("camera");
    namedWindow("qrcode");

    Mat frame;
    while(cap.read(frame)) {
        imshow("camera", frame);
        if(detector.drawQRCode(frame)) {
            auto &&qr = detector.getImage();
            imshow("qrcode", qr);
        }
        if(waitKey(30) >= 0)
            break;
    }

    destroyAllWindows();
}
