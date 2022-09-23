#ifndef SCREENCAMERA_H
#define SCREENCAMERA_H

#include <QThread>
#include <QObject>
#include <QScreen>
#include <QTimer>

class ScreenCamera;

/**
 * @brief The ScreenCameraInfo class 查询屏幕相机的信息的类
 * 用这个类可以查到当前计算机接了几个显示器，给出每个显示器的名称（deviceName）
 */
class ScreenCameraInfo
{
    friend class ScreenCamera;
public:
    explicit ScreenCameraInfo(QScreen * screen = nullptr);
    ScreenCameraInfo(const ScreenCameraInfo &info);
    /**
     * @brief description 没有实现，返回空字符串
     * @return 返回空字符串。
     */
    QString description() const;
    /**
     * @brief deviceName 给出屏幕的名称，比如 "\\\\.\\DISPLAY1", "VGA1" 一类的名称。
     * @return
     */
    QString deviceName() const;
    bool isNull() const;
    bool operator!=(const ScreenCameraInfo &other) const;
    ScreenCameraInfo &operator=(const ScreenCameraInfo &other);
    bool operator==(const ScreenCameraInfo &other) const;

    static QList<ScreenCameraInfo> availableCameras();
    static ScreenCameraInfo defaultCamera();

private:
    QScreen* m_screen;
};

/**
 * @brief The ScreenCamera class 将截屏功能封装成一个相机，可以像相机一样的操作。
 */
class ScreenCamera : public QThread
{
    Q_OBJECT
public:
    explicit ScreenCamera(QObject *parent = 0);
    explicit ScreenCamera(ScreenCameraInfo &info, QObject *parent = 0);
    /**
     * @brief openDevice 打开一个特定的截屏相机，当计算机连接了多个显示器时通过这个 info 指定截取某一个屏幕的图像
     * @param info
     * @return
     */
    bool openDevice(const ScreenCameraInfo &info);
    /**
     * @brief setFrameRate 设置定时截屏的帧率。这里模仿相机的帧率设置。
     * @param fps  frame per second 每秒的截屏数量。 [0.001 - 50]
     */
    void setFrameRate(double fps);
    /**
     * @brief setRoi 设置感兴趣的区域，利用这个函数我们可以只截取屏幕的一部分区域
     * @param x 截屏区域的起始X坐标
     * @param y 截屏区域的起始Y坐标
     * @param width 截屏区域的宽度，-1 表示完整的屏幕宽度
     * @param height 截屏区域的高度，-1 表示完整的屏幕高度
     */
    void setRoi(int x, int y, int width, int height);
    void run() override;
    /**
     * @brief capture 获取截屏图像
     * @return
     */
    QImage capture();
signals:
    /**
     * @brief imageChanged 定时截取图像时没截取一帧图像就激发一次这个信号
     * @param image
     */
    void imageChanged(const  QImage& image);
public slots:
    /**
     * @brief start 开始图像采集
     */
    void start();
    /**
     * @brief stop 停止图像采集
     */
    void stop();

private:
    int m_sleepInterval;
    QScreen* m_screen;
    int m_x;
    int m_y;
    int m_width;
    int m_height;
};

#endif // SCREENCAMERA_H
