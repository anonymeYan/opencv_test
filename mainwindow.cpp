#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QThread* sub = new QThread;
    capture* cap = new capture;
    cap->moveToThread(sub);
    sub->start();
    connect(ui->pushButton_2, &QPushButton::clicked, cap, &capture::working);
    connect(ui->pushButton_3, &QPushButton::clicked, cap, &capture::gray);
    connect(ui->pushButton_4,&QPushButton::clicked,cap,&capture::harrisDetection);
    connect(ui->pushButton_5,&QPushButton::clicked,cap,&capture::siftDetection);
    connect(ui->pushButton_6,&QPushButton::clicked,cap,&capture::TMatch);
    connect(ui->pushButton_7,&QPushButton::clicked,cap,&capture::orbDetection);
    connect(ui->pushButton_8,&QPushButton::clicked,cap,&capture::BfMatch);
    connect(ui->pushButton_9,&QPushButton::clicked,cap,&capture::FlannMatch);
    connect(ui->pushButton_10,&QPushButton::clicked,cap,&capture::SiftFlann);
    connect(ui->pushButton_11,&QPushButton::clicked,cap,&capture::OrbFlann);
    connect(ui->pushButton_12,&QPushButton::clicked,cap,&capture::SiftBf);
    connect(ui->pushButton_13,&QPushButton::clicked,cap,&capture::orbBf);
    connect(ui->pushButton_14,&QPushButton::clicked,cap,&capture::TMultiMatch);
    connect(this,SIGNAL(sendMat(const Mat&)),cap,SLOT(sendMatSlot(const Mat&)));
    connect(cap,SIGNAL(imageChanged(const Mat&)),this,SLOT(imageChangedSlot(const Mat&)));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    QString file = QFileDialog::getOpenFileName(this,"选择图片","","Images(*.png *.bmp *.jpg)");
    if(file.isEmpty()){
        return;
    }

    src = cv::imread(file.toLocal8Bit().toStdString());
    if(src.empty()){
        return;
    }
    emit sendMat(src);
    //    cv::imshow("vision123",src);
    //    cv::waitKey(0);
    //    cv::destroyAllWindows();

    QImage img = cvMat2QImage(src);
    QPixmap tempPixmap = QPixmap::fromImage(img);
    QPixmap fitPixmap = tempPixmap.scaled(ui->label->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
    ui->label->setPixmap(fitPixmap);


}


//void MainWindow::on_pushButton_3_clicked()
//{
//    if(src.empty()){
//        return;
//    }
//    Mat gray;
//    cvtColor(src,gray,COLOR_BGR2GRAY);
//    QImage img = cvMat2QImage(gray);
//    QPixmap tempPixmap = QPixmap::fromImage(img);
//    QPixmap fitPixmap = tempPixmap.scaled(ui->label->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
//    ui->label_2->setPixmap(fitPixmap);
//}



void MainWindow::imageChangedSlot(const Mat &src)
{
    QImage img = cvMat2QImage(src);
    QPixmap tempPixmap = QPixmap::fromImage(img);
    QPixmap fitPixmap = tempPixmap.scaled(ui->label->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation);
    ui->label->setPixmap(fitPixmap);
}



QImage MainWindow::cvMat2QImage(const cv::Mat & mat)
{
    // 8-bits unsigned, NO. OF CHANNELS = 1
    if (mat.type() == CV_8UC1)
    {
        QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
        // Set the color table (used to translate colour indexes to qRgb values)
        image.setColorCount(256);
        for (int i = 0; i < 256; i++)
        {
            image.setColor(i, qRgb(i, i, i));
        }
        // Copy input Mat
        uchar *pSrc = mat.data;
        for (int row = 0; row < mat.rows; row++)
        {
            uchar *pDest = image.scanLine(row);
            memcpy(pDest, pSrc, mat.cols);
            pSrc += mat.step;
        }
        return image;
    }
    // 8-bits unsigned, NO. OF CHANNELS = 3
    else if (mat.type() == CV_8UC3)
    {
        // Copy input Mat
        const uchar *pSrc = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped();
    }
    else if (mat.type() == CV_8UC4)
    {
        qDebug() << "CV_8UC4";
        // Copy input Mat
        const uchar *pSrc = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        return image.copy();
    }
    else
    {
        qDebug() << "ERROR: Mat could not be converted to QImage.";
        return QImage();
    }
}
cv::Mat MainWindow::QImage2cvMat(QImage image)
{
    cv::Mat mat;
    switch (image.format())
    {
    case QImage::Format_ARGB32:
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32_Premultiplied:
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
        break;
    case QImage::Format_RGB888:
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
        break;
    case QImage::Format_Indexed8:
        mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
        break;
    }
    return mat;
}






