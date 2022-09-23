#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "opencv2/opencv.hpp"
#include "qfiledialog.h"
#include "qimage.h"
#include "qdebug.h"
#include "qthread.h"
#include "capture.h"
#include "qevent.h"


using namespace cv;

namespace Ui {
class MainWindow;
}



typedef int(*pDD_btn)(int btn);
typedef int(*pDD_whl)(int whl);
typedef int(*pDD_key)(int keycode, int flag);
typedef int(*pDD_mov)(int x, int y);
typedef int(*pDD_str)(char *str);
typedef int(*pDD_todc)(int vk);
typedef int(*pDD_movR)(int dx, int dy);


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    QImage cvMat2QImage(const cv::Mat & mat);
    cv::Mat QImage2cvMat(QImage image);

    void ddTest();

    pDD_btn      DD_btn;          //Mouse button
    pDD_whl      DD_whl;		     //Mouse wheel
    pDD_key      DD_key;		     //Mouse move abs.
    pDD_mov    DD_mov;		 //Mouse move rel.
    pDD_str       DD_str;			 //Keyboard
    pDD_todc    DD_todc;		 //Input visible char
    pDD_movR   DD_movR;	     //VK to ddcode



signals:
    void sendMat(const Mat& image);

public slots:
    void imageChangedSlot(const Mat& src);

private slots:
    void on_pushButton_clicked();



    void on_pushButton_15_clicked();

    void on_pushButton_16_clicked();

private:
    Ui::MainWindow *ui;
    Mat src;

};

#endif // MAINWINDOW_H
