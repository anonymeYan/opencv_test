#-------------------------------------------------
#
# Project created by QtCreator 2022-09-20T15:53:19
#
#-------------------------------------------------

QT       += core gui

LIBS += -lGdi32
LIBS += -luser32

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = opencv002
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

SOURCES += \
        capture.cpp \
        main.cpp \
        mainwindow.cpp

HEADERS += \
        capture.h \
        mainwindow.h

FORMS += \
        mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target



RESOURCES += \
    res.qrc







win32:CONFIG(release, debug|release): LIBS += -LD:/Qt/newbuild/install/x64/vc15/lib/ -lopencv_world454
else:win32:CONFIG(debug, debug|release): LIBS += -LD:/Qt/newbuild/install/x64/vc15/lib/ -lopencv_world454d
else:unix: LIBS += -LD:/Qt/newbuild/install/x64/vc15/lib/ -lopencv_world454

INCLUDEPATH += D:/Qt/newbuild/install/include
DEPENDPATH += D:/Qt/newbuild/install/include
