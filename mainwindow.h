#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "opencv2/opencv.hpp"
#include "qfiledialog.h"
#include "qimage.h"
#include "qdebug.h"
#include "qthread.h"
#include "capture.h"

using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    QImage cvMat2QImage(const cv::Mat & mat);
    cv::Mat QImage2cvMat(QImage image);
signals:
    void sendMat(const Mat& image);

public slots:
    void imageChangedSlot(const Mat& src);

private slots:
    void on_pushButton_clicked();



private:
    Ui::MainWindow *ui;
    Mat src;

};

#endif // MAINWINDOW_H
