#ifndef CAPTURE_H
#define CAPTURE_H

#include <QObject>

#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/mat.inl.hpp"

#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"


#include <Windows.h>
#include "qfiledialog.h"
#include "qimage.h"
#include "qdebug.h"
#include "qthread.h"
#include "qtimer.h"
#include "QTime"
#include "qeventloop.h"
#include "QtMath"



using namespace cv;


typedef int(*pDD_btn)(int btn);
typedef int(*pDD_whl)(int whl);
typedef int(*pDD_key)(int keycode, int flag);
typedef int(*pDD_mov)(int x, int y);
typedef int(*pDD_str)(char *str);
typedef int(*pDD_todc)(int vk);
typedef int(*pDD_movR)(int dx, int dy);

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
    void SiftBf();
    void SiftFlann();
    void surfFlann();
    void orbBf();
    void OrbFlann();

    void preWork();
    void colorDetc();

    void TMatch();
    void TMultiMatch();

    QPoint calSecondKeg(const QPoint& pt1,const QPoint& pt2);
    QPoint calSecondKeg1(const QPoint& pt1,const QPoint& pt2);
    void calDistance(const QPoint& pt1,const QPoint& pt2);
    void Gp();
    void GpKeg();
    void GpEnemy();
    void GpTest();
    void GpDetcSiftFlann();
    void GpDetcSurfFlann();


    void enemyMatch();

    void tsMatch();


    pDD_btn      DD_btn;          //Mouse button
    pDD_whl      DD_whl;		     //Mouse wheel
    pDD_key      DD_key;		     //Mouse move abs.
    pDD_mov    DD_mov;		 //Mouse move rel.
    pDD_str       DD_str;			 //Keyboard
    pDD_todc    DD_todc;		 //Input visible char
    pDD_movR   DD_movR;	     //VK to ddcode

signals :
    void imageChanged(const Mat& src);

public slots :
    void sendMatSlot(const Mat& image);

private:
    BITMAPINFOHEADER createBitmapHeader(int width, int height);
    Mat captureScreenMat(HWND hwnd);

    Mat capMat;

    Mat keg;
    Ptr<xfeatures2d::SURF> surf_detector;
    std::vector<KeyPoint> keg_kp;
    Mat keg_des;
};

#endif // CAPTURE_H
