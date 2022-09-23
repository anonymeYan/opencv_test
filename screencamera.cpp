#include "ScreenCamera.h"
#include <QApplication>
#include <QDesktopWidget>
#include <QDebug>

ScreenCameraInfo::ScreenCameraInfo(const ScreenCameraInfo &info)
    :m_screen(nullptr)
{
    m_screen = info.m_screen;
}

ScreenCameraInfo::ScreenCameraInfo(QScreen * screen)
    :m_screen(nullptr)
{
    m_screen = screen;
}

bool ScreenCameraInfo::isNull() const
{
    return (m_screen == nullptr);
}

QString ScreenCameraInfo::description() const
{
    return "";
}
QString ScreenCameraInfo::deviceName() const
{
    if(m_screen == nullptr) return "";
    return m_screen->name();
}

ScreenCameraInfo ScreenCameraInfo::defaultCamera()
{
    QScreen * w = QGuiApplication::primaryScreen();
    return ScreenCameraInfo(w);
}

QList<ScreenCameraInfo> ScreenCameraInfo::availableCameras()
{
    QList<QScreen *> list = QGuiApplication::screens();
    QList<ScreenCameraInfo> info;
    foreach (QScreen * s, list)
    {
        ScreenCameraInfo i(s);
        info.append(i);
    }

    return info;
}

bool ScreenCameraInfo::operator!=(const ScreenCameraInfo &other) const
{
    return (m_screen != other.m_screen);
}

ScreenCameraInfo & ScreenCameraInfo::operator=(const ScreenCameraInfo &other)
{
    m_screen = other.m_screen;
    return *this;
}

bool ScreenCameraInfo::operator==(const ScreenCameraInfo &other) const
{
    return m_screen == other.m_screen;
}


ScreenCamera::ScreenCamera(ScreenCameraInfo &info, QObject *parent) : QThread(parent)
{
    m_screen = info.m_screen;
    m_sleepInterval = 500;
    m_x = 0;
    m_y = 0;
    m_width = -1;
    m_height = -1;
}

ScreenCamera::ScreenCamera(QObject *parent) : QThread(parent)
{
    m_screen = nullptr;
    m_sleepInterval = 500;
    m_x = 0;
    m_y = 0;
    m_width = -1;
    m_height = -1;
}

bool ScreenCamera::openDevice(const ScreenCameraInfo &info)
{
    m_screen = info.m_screen;
    return (m_screen);
}

void ScreenCamera::setFrameRate(double fps)
{
    fps = qBound(0.001, fps, 50.0);
    m_sleepInterval = 1000 / fps;
}

void ScreenCamera::setRoi(int x, int y, int width, int height)
{
    if(m_screen)
    {
        QRect rect = m_screen->geometry();

        m_x = qBound(0, x, rect.width());
        m_y = qBound(0, y, rect.height());
        m_width = qBound(-1, width, rect.width());
        m_height = qBound(-1, height, rect.height());
    }
}

void ScreenCamera::run()
{
    while(1)
    {   qDebug() << "444 ";
        if(m_screen)
        {
            QDesktopWidget *desk = QApplication::desktop();
            //qDebug() << "m_width = " << m_width;
            //qDebug() << "m_height = " << m_height;
            QPixmap p = m_screen->grabWindow(desk->winId(), m_x, m_y, m_width, m_height);
            QImage image = p.toImage();
            emit imageChanged(image);
        }
        msleep(m_sleepInterval);
    }
}

QImage ScreenCamera::capture()
{
    if(m_screen)
    {
        QDesktopWidget *desk = QApplication::desktop();
        QPixmap p = m_screen->grabWindow(desk->winId(), m_x, m_y, m_width, m_height);
        QImage image = p.toImage();
        return image;
    }
    return QImage();
}

void ScreenCamera::stop()
{
    terminate();
}

void ScreenCamera::start()
{
    QThread::start(QThread::InheritPriority);
}
