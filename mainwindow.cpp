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
    connect(ui->pushButton_16,&QPushButton::clicked,cap,&capture::GpKeg);
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

void MainWindow::ddTest()
{
    HMODULE hDll = LoadLibraryW(L"D:\\Qt\\DD94687.64.dll");
    if (hDll == nullptr)
    {
        qDebug() << "ERROR:-1";
        return ;
    }

    DD_btn = (pDD_btn)GetProcAddress(hDll, "DD_btn");
    DD_whl = (pDD_whl)GetProcAddress(hDll, "DD_whl");
    DD_key = (pDD_key)GetProcAddress(hDll, "DD_key");
    DD_mov = (pDD_mov)GetProcAddress(hDll, "DD_mov");
    DD_str = (pDD_str)GetProcAddress(hDll, "DD_str");
    DD_todc = (pDD_todc)GetProcAddress(hDll, "DD_todc");
    DD_movR = (pDD_movR)GetProcAddress(hDll, "DD_movR");

    if (!(DD_btn && DD_whl && DD_key && DD_mov && DD_str  && DD_todc && DD_movR))
    {
        qDebug() << "ERROR:-2";
        return ;
    }

    int st = DD_btn(0);
    if (st != 1)
    {
        //DD Initialize Error
        qDebug() << "ERROR:"<<st;
        return ;
    }

    qDebug()  << "Mouse R.click";
    system("pause");
    //1==L.down, 2==L.up, 4==R.down, 8==R.up, 16==M.down, 32==M.up
    DD_btn(4);
    Sleep(50); //may, delay 50ms
    DD_btn(8);

    qDebug()  << "Mouse Move rel.";
    system("pause");
    DD_movR(20, 20);   //move rel.
    qDebug()  << "Mouse Move abs.";
    system("pause");
    DD_mov(200, 200); //move abs.

    qDebug()  << "Keyboard L.win";
    system("pause");
    int ddcode = 601;		//Left.win == 601 in ddcode
    ddcode = DD_todc(VK_LWIN);
    DD_key(ddcode, 1);
    Sleep(1);					//may, delay 50ms
    DD_key(ddcode, 2);

    //    qDebug()  << "type visiable char";
    //    system("pause");
    //    DD_str("Keyboard char [A-Za_z] {@$} ");

    qDebug()  << "ctrl+alt+del";
    system("pause");
    DD_key(600, 1);  //600 == L.CTRL down
    DD_key(602, 1);  //602 == L.ALT   down
    DD_key(706, 1);  //706 == DEL   down
    DD_key(706, 2);
    DD_key(602, 2); 	 //up
    DD_key(600, 2);

    FreeLibrary(hDll);

    qDebug() << "1";
    return ;
}




void MainWindow::on_pushButton_15_clicked()
{
    ddTest();
}


//void MainWindow::keyPressEvent(QKeyEvent *event){
//    switch(event->key()){
//    case Qt::Key_Escape:
//        ui->textEdit_press->append("Key_Escape Press");
//        break;
//    case Qt::Key_Tab:
//        ui->textEdit_press->append("Key_Tab Press");
//        break;
//    case Qt::Key_Enter:
//        ui->textEdit_press->append("Key_Enter Press");
//        break;
//    case Qt::Key_Delete:
//        ui->textEdit_press->append("Key_Delete Press");
//        break;
//    case Qt::Key_Space:
//        ui->textEdit_press->append("Key_Space Press");
//        break;
//    case Qt::Key_Left:
//        ui->textEdit_press->append("Key_Left Press");
//        break;
//    case Qt::Key_Up:
//        ui->textEdit_press->append("Key_Up Press");
//        break;
//    case Qt::Key_Right:
//        ui->textEdit_press->append("Key_Right Press");
//        break;
//    case Qt::Key_Down:
//        ui->textEdit_press->append("Key_Down Press");
//        break;
//    case Qt::Key_Q:
//        ui->textEdit_press->append("q Press");
//        break;
//    case Qt::Key_W:
//        ui->textEdit_press->append("w Press");
//        break;
//    case Qt::Key_E:
//        ui->textEdit_press->append("e Press");
//        break;
//    case Qt::Key_R:
//        ui->textEdit_press->append("r Press");
//        break;
//    case Qt::Key_1:
//        ui->textEdit_press->append("1 Press");
//        break;
//    case Qt::Key_2:
//        ui->textEdit_press->append("1 Press");
//        break;
//    case Qt::Key_3:
//        ui->textEdit_press->append("1 Press");
//        break;
//        /*default:
//      this->ui.textEdit->append("KeyEvent");*/
//    }
//}



void MainWindow::on_pushButton_16_clicked()
{
    static int flag = 0;
    if(flag == 0){
        ui->pushButton_16->setText("GP_open");
        flag=1;
    }
    else {
        ui->pushButton_16->setText("GP_close");
        flag=0;
    }
}
