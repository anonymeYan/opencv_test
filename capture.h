#ifndef CAPTURE_H
#define CAPTURE_H

#include <QObject>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"

#include <Windows.h>
#include "qfiledialog.h"
#include "qimage.h"
#include "qdebug.h"
#include "qthread.h"



using namespace cv;




class capture : public QObject
{
    Q_OBJECT
public:
    explicit capture(QObject *parent = nullptr);

    void working();
    void gray();
    void harrisDetection();
    void siftDetection();
    void surfDetection();
    void orbDetection();
    void BfMatch();
    void FlannMatch();

    void TMatch();
    void TMultiMatch();
    void SiftBf();
    void SiftFlann();
    void orbBf();
    void OrbFlann();

signals :
    void imageChanged(const Mat& src);

public slots :
    void sendMatSlot(const Mat& image);

private:
    BITMAPINFOHEADER createBitmapHeader(int width, int height);
    Mat captureScreenMat(HWND hwnd);

    Mat capMat;

};

#endif // CAPTURE_H
