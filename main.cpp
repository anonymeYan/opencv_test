#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    qRegisterMetaType< Mat >("Mat");
    qRegisterMetaType< Mat >("Mat&");

    MainWindow w;
    w.show();

    return a.exec();
}
