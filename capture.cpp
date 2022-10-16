#include "capture.h"

static int thresh = 130;
static int max_count = 255;
static Mat img, img_gray;
static const char* output_title = "Harris Corner Dectction Result";
static void Harris_Demo(int, void *);

static void thresh_Demo(int, void *);
static Mat input;
void thresh_Demo(int, void *) {
    if(input.empty()){
        return  ;
    }
    cv::Mat src = input.clone();
    //预处理
    Mat src_down;
    pyrDown(src,src_down);
    Mat src_gray;
    cvtColor(src_down,src_gray,COLOR_RGB2GRAY);
    Mat src_thresh;
    cv::threshold(src_gray,src_thresh,thresh,255,THRESH_BINARY);
    imshow("threshold result", src_thresh);
}



void Harris_Demo(int, void *) {

    Mat dst, norm_dst, normScaleDst;
    dst = Mat::zeros(img_gray.size(), CV_32FC1);
    //harris角点核心函数
    int blockSize = 2;
    int ksize = 3;
    int k = 0.04;

    cornerHarris(img_gray, dst, blockSize, ksize, k, BORDER_DEFAULT);
    //上述输出的取值范围并不是0-255 需要按照最大最小值进行归一化
    normalize(dst, norm_dst, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(norm_dst, normScaleDst);

    Mat resultImg = img.clone();
    //用彩色来显示
    for (int row = 0; row < resultImg.rows; row++) {
        //定义每一行的指针
        uchar* currentRow = normScaleDst.ptr(row);
        for (int col = 0; col < resultImg.cols; col++) {
            int value = (int)*currentRow;
            if (value > thresh) {
                circle(resultImg, Point(col, row), 2, Scalar(0, 0, 255), 2, 8, 0);
            }
            currentRow++;
        }
    }

    imshow(output_title, resultImg);
}




capture::capture(QObject *parent):QObject (parent)
{
    HMODULE hDll = LoadLibraryW(L"D:\\Qt\\DDHID64.dll");
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

    qDebug()  << "DD load success!";


    keg = imread("F:\\CODE_PROJECT\\opencv002\\res\\lolkeg1.png");
    cvtColor(keg,keg,COLOR_RGB2GRAY);
    surf_detector = xfeatures2d::SURF::create(1000);
    surf_detector->detectAndCompute(keg,Mat(),keg_kp,keg_des);
    qDebug()<<"Total keyponits: "<<keg_kp.size();

}

void capture::working()
{
    qDebug() << "当前线程对象的地址: " << QThread::currentThread();
    HWND hwnd = GetDesktopWindow();
    Mat img = captureScreenMat(hwnd);
    emit imageChanged(img);
    imshow("screenshot",img);
    waitKey(0);
    destroyAllWindows();
}

void capture::gray()
{
    if(capMat.empty()){
        return  ;
    }
    Mat src = capMat.clone();
    Mat gray;
    cvtColor(src,gray,COLOR_BGR2GRAY);
    imshow("SIFT KeyPoints",gray);
    imwrite("gray.png",gray);
    waitKey(0);

}

void capture::harrisDetection()
{
    if(capMat.empty()){
        return  ;
    }
    img = capMat.clone();
    namedWindow("input image", WINDOW_AUTOSIZE);
    imshow("input image", capMat);
    //以上是图像处理的标准开头
    namedWindow(output_title, WINDOW_AUTOSIZE);
    cvtColor(img, img_gray, COLOR_BGR2GRAY);

    createTrackbar("Threshold", output_title, &thresh, max_count, Harris_Demo);
    Harris_Demo(0, 0);

    waitKey(0);

}

void capture::siftDetection()
{
    QTime time;
    time.start();
    if(capMat.empty()){
        return  ;
    }
    Mat src = capMat.clone();
    Mat gray;
    cvtColor(capMat,gray,COLOR_BGR2GRAY);
    Ptr<SIFT> sift = SIFT::create(300);
    std::vector<KeyPoint> kp;
    Mat des;
    sift->detectAndCompute(gray,Mat(),kp,des);
    qDebug()<<"Total keyponits: "<<kp.size();
    qDebug()<<"time= "<< time.elapsed()/1000.0<<"s";

    Mat kpImage;
    drawKeypoints(gray,kp,kpImage,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
    imshow("SIFT KeyPoints",kpImage);
    waitKey(0);

}

void capture::surfDetection()
{
    QTime time;
    time.start();
    if(capMat.empty()){
        return  ;
    }
    Mat src = capMat.clone();
    Mat gray;
    cvtColor(src,gray,COLOR_BGR2GRAY);

    Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(300);
    std::vector<KeyPoint> kp;
    Mat des;
    detector->detectAndCompute(gray,Mat(),kp,des);
    qDebug()<<"Total keyponits: "<<kp.size();
    qDebug()<<"time= "<< time.elapsed()/1000.0<<"s";

    Mat kpImage;
    drawKeypoints(gray,kp,kpImage,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
    imshow("SURF KeyPoints",kpImage);
    waitKey(0);

}


void capture::orbDetection()
{
    QTime time;
    time.start();
    if(capMat.empty()){
        return  ;
    }
    Ptr<ORB> orb = ORB::create(300);
    Mat gray,kp_img;
    cvtColor(capMat,gray,COLOR_BGR2GRAY);
    std::vector<KeyPoint> kp;
    Mat des;
    orb->detectAndCompute(gray,Mat(),kp,des);
    qDebug()<<"Total keyponits: "<<kp.size();
    qDebug()<<"time= "<< time.elapsed()/1000.0<<"s";

    drawKeypoints(gray,kp,kp_img);
    imshow("kp_img", kp_img);
    waitKey(0);
}





BITMAPINFOHEADER capture::createBitmapHeader(int width, int height)
{
    BITMAPINFOHEADER  bi;

    // create a bitmap
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = width;
    bi.biHeight = -height;  //this is the line that makes it draw upside down or not
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;
    bi.biSizeImage = 0;
    bi.biXPelsPerMeter = 0;
    bi.biYPelsPerMeter = 0;
    bi.biClrUsed = 0;
    bi.biClrImportant = 0;

    return bi;
}

Mat capture::captureScreenMat(HWND hwnd)
{
    Mat src;

    // get handles to a device context (DC)
    HDC hwindowDC = GetDC(hwnd);
    HDC hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
    SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);

    // define scale, height and width
    int screenx = GetSystemMetrics(SM_XVIRTUALSCREEN);
    int screeny = GetSystemMetrics(SM_YVIRTUALSCREEN);
    int width = GetSystemMetrics(SM_CXVIRTUALSCREEN);
    int height = GetSystemMetrics(SM_CYVIRTUALSCREEN);

    // create mat object
    src.create(height, width, CV_8UC4);

    // create a bitmap
    HBITMAP hbwindow = CreateCompatibleBitmap(hwindowDC, width, height);
    BITMAPINFOHEADER bi = createBitmapHeader(width, height);

    // use the previously created device context with the bitmap
    SelectObject(hwindowCompatibleDC, hbwindow);

    // copy from the window device context to the bitmap device context
    StretchBlt(hwindowCompatibleDC, 0, 0, width, height, hwindowDC, screenx, screeny, width, height, SRCCOPY);  //change SRCCOPY to NOTSRCCOPY for wacky colors !
    GetDIBits(hwindowCompatibleDC, hbwindow, 0, height, src.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);            //copy from hwindowCompatibleDC to hbwindow

    // avoid memory leak
    DeleteObject(hbwindow);
    DeleteDC(hwindowCompatibleDC);
    ReleaseDC(hwnd, hwindowDC);

    return src;
}


void capture::TMatch()
{
    if(capMat.empty()){
        return  ;
    }
    QTime time;
    time.start();
    cv::Mat src = capMat.clone();
    cv::Mat sample = imread("F:\\CODE_PROJECT\\opencv002\\res\\lolkegblack.png");
    if(sample.empty()){
        qDebug() <<"sample is empty";
        return  ;
    }
    int col= sample.cols;
    int row= sample.rows;
    qDebug() <<"sample.cols:"<<col<<" sample.rows:"<<row;
    qDebug()<<"src.type:"<<src.type()<<"sample.type:"<<sample.type();
    qDebug()<<"time0= "<< time.elapsed()/1000.0<<"s";
    //预处理
    Mat src_down,sample_down;
    pyrDown(src,src_down);
    pyrDown(sample,sample_down);
    //    Mat src_gray,sample_gray;
    //    cvtColor(src_down,src_gray,COLOR_RGB2GRAY);
    //    cvtColor(sample_down,sample_gray,COLOR_RGB2GRAY);
    //    Mat src_thresh,sample_thresh;
    //    cv::threshold(src_down,src_thresh,85,255,THRESH_BINARY);
    //    cv::threshold(sample_down,sample_thresh,85,255,THRESH_BINARY);
    // 匹配
    cv::Mat result;
    matchTemplate(src, sample, result, TM_CCOEFF_NORMED);//1为最匹配
    //matchTemplate(src, sample, result, TM_SQDIFF);//0为最匹配
    qDebug()<<"time1= "<< time.elapsed()/1000.0<<"s";
    //获取单个目标
    double minValue; double maxValue; Point minLocation; Point maxLocation;
    Point matchLocation;
    minMaxLoc(result, &minValue, &maxValue, &minLocation, &maxLocation, Mat());
    qDebug()<<"time2= "<< time.elapsed()/1000.0<<"s";
    matchLocation = maxLocation;
    qDebug()<<"minVal"<<minValue;
    qDebug()<<"maxVal"<<maxValue;
    qDebug()<<"maxLoc("<<maxLocation.x << ","<<maxLocation.y<<")";
    if(maxValue<0.6){
        qDebug()<<"match failed";
        return;
    }
    // 框选结果
    cv::Mat draw = src.clone();
    Point matchF;
    matchF.x = matchLocation.x;
    matchF.y = matchLocation.y;
    rectangle(draw, matchF, Point(matchF.x + sample.cols, matchF.y + sample.rows), Scalar(0, 0, 255), 2, 8, 0);
    qDebug()<<"total_time= "<< time.elapsed()/1000.0<<"s";
    imshow("draw", draw);
    waitKey(0);
}


void capture::TMultiMatch()
{
    QTime time;
    time.start();
    if(capMat.empty()){
        return  ;
    }
    cv::Mat src = capMat.clone();
    cv::Mat sample = imread("F:\\CODE_PROJECT\\opencv002\\res\\lolkeg1.png");
    if(sample.empty()){
        qDebug() <<"sample is empty";
        return  ;
    }
    //预处理
    //    Mat src_gauss,sample_gauss;
    //    GaussianBlur(src, src_gauss, Size(3, 3), 0);
    //    GaussianBlur(sample, sample_gauss, Size(3, 3), 0);
    Mat src_down,sample_down;
    pyrDown(src,src_down);
    pyrDown(sample,sample_down);
    //    Mat src_gray,sample_gray;
    //    cvtColor(src_down,src_gray,COLOR_BGR2GRAY);
    //    cvtColor(sample_down,sample_gray,COLOR_BGR2GRAY);
    //    Mat src_thresh,sample_thresh;
    //    cv::threshold(src_down,src_thresh,90,255,THRESH_BINARY);
    //    cv::threshold(sample_down,sample_thresh,90,255,THRESH_BINARY);

    //匹配
    Mat result;
    matchTemplate(src_down, sample_down, result, TM_CCORR_NORMED);
    qDebug()<<"(sample_x="<<sample.cols<<",sample_y="<<sample.rows<<")";
    qDebug()<<"time0= "<< time.elapsed()/1000.0<<"s";
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    qDebug()<<"maxVal"<<maxVal;
    qDebug()<<"maxLoc("<<maxLoc.x << ","<<maxLoc.y<<")";
    qDebug()<<"time1= "<< time.elapsed()/1000.0<<"s";
    //多目标匹配
    QList<QPoint> listResult;
    for (int i = 0; i < result.rows; i++)
    {
        for (int j = 0; j < result.cols; j++)
        {
            double val = result.at<float>(i, j);//灰度值

            //3*3邻域非极大值抑制
            if (val > 0.9)
            {
                if(i!=0 && j!=0 && i!=(result.rows-1) && j!=(result.cols-1)){
                    //当前像素的灰度值大于阈值且该像素是其3*3邻域最值时，判定其为目标
                    if (result.at<float>(i - 1, j - 1) < val &&
                            result.at<float>(i - 1, j) < val &&
                            result.at<float>(i - 1, j + 1) < val &&
                            result.at<float>(i, j - 1) < val &&
                            result.at<float>(i, j + 1) < val &&
                            result.at<float>(i + 1, j - 1) < val &&
                            result.at<float>(i + 1, j) < val &&
                            result.at<float>(i + 1, j + 1) < val)
                    {
                        qDebug()<<"result="<<val;
                        qDebug()<<"(x="<<j<<",y="<<i<<")";
                        listResult.append(QPoint(j*2, i*2));
                        //结果绘制
                        rectangle(src_down, Rect(j, i, sample.cols/2, sample.rows/2),Scalar(0, 255, 0), 2);

                        char text[10];
                        float score = result.at<float>(i, j);
                        sprintf_s(text, "%.2f",score);
                        putText(src_down, text, Point(j, i), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
                    }
                }
            }
        }
    }
    qDebug()<<"total time= "<< time.elapsed()/1000.0<<"s";
    qDebug()<<"total result= "<< listResult.size();
    if(listResult.size()>1){
        calDistance(listResult.at(0),listResult.at(1));
    }
    imshow("multi template match result", src_down);
    waitKey(0);
}


QPoint capture::calSecondKeg(const QPoint &pt1, const QPoint &pt2)
{
    double x1 = pt1.x();
    double y1 = pt1.y();
    double x2 = pt2.x();
    double y2 = pt2.y();
    double x3 = 0.0, y3 = 0.0;
    qDebug() <<"x1:"<<pt1.x()<<"y1:"<<pt1.y();
    qDebug() <<"x2:"<<pt2.x()<<"y2:"<<pt2.y();

    x3 = x1 + 330 * cos(double(atan2(y2 - y1, x2 - x1)));
    y3 = y1 + 330 * sin(double(atan2(y2 - y1, x2 - x1)));

    double x = x1 - x3;
    double y = y1 - y3;
    double dis = sqrt(x*x+y*y);
    qDebug() <<"x3:"<<x3<<"y3:"<<y3;
    qDebug() <<"dis:"<<dis;
    return QPoint(x3, y3);
}

QPoint capture::calSecondKeg1(const QPoint &pt1, const QPoint &pt2)
{
    double x1 = pt1.x();
    double y1 = pt1.y();
    double x2 = pt2.x();
    double y2 = pt2.y();
    double x3 = 0.0, y3 = 0.0;
    qDebug() <<"x1:"<<pt1.x()<<"y1:"<<pt1.y();
    qDebug() <<"x2:"<<pt2.x()<<"y2:"<<pt2.y();
    double angle = atan2(y2 - y1, x2 - x1)*180/M_PI;//-180~180度 逆时针度数
    double result = -angle;
    if(result<0)result = result + 360;
    qDebug()<<"result angle"<<result;
    double distance;
    int compare = result;
    if(compare==0 || compare == 180)
    {
        distance = 410;
        x3 = x1 + distance * cos(double(atan2(y2 - y1, x2 - x1)));
        y3 = y1 + distance * sin(double(atan2(y2 - y1, x2 - x1)));
    }else if (compare>0 && compare<=30) {
        distance = 340 + (410-340)*(30-result)/30;
        x3 = x1 + distance * cos(double(atan2(y2 - y1, x2 - x1)));
        y3 = y1 + distance * sin(double(atan2(y2 - y1, x2 - x1)));
    }else if (compare>30 && compare<=60){
        distance = 285 + (340-285)*(60-result)/30;
        x3 = x1 + distance * cos(double(atan2(y2 - y1, x2 - x1)));
        y3 = y1 + distance * sin(double(atan2(y2 - y1, x2 - x1)));
    }else if (compare>60 && compare<=120){
        distance = 285;
        x3 = x1 + distance * cos(double(atan2(y2 - y1, x2 - x1)));
        y3 = y1 + distance * sin(double(atan2(y2 - y1, x2 - x1)));
    }else if (compare>120 && compare<=150){
        distance = 285 + (340-285)*(result-120)/30;
        x3 = x1 + distance * cos(double(atan2(y2 - y1, x2 - x1)));
        y3 = y1 + distance * sin(double(atan2(y2 - y1, x2 - x1)));
    }else if (compare>150 && compare<180){
        distance = 340 + (410-340)*(result-150)/30;
        x3 = x1 + distance * cos(double(atan2(y2 - y1, x2 - x1)));
        y3 = y1 + distance * sin(double(atan2(y2 - y1, x2 - x1)));
    }else if (compare>180 && compare<=210){
        distance = 380 + (410-380)*(210-result)/30;
        x3 = x1 + distance * cos(double(atan2(y2 - y1, x2 - x1)));
        y3 = y1 + distance * sin(double(atan2(y2 - y1, x2 - x1)));
    }else if (compare>210 && compare<=240){
        distance = 335 + (380-335)*(240-result)/30;
        x3 = x1 + distance * cos(double(atan2(y2 - y1, x2 - x1)));
        y3 = y1 + distance * sin(double(atan2(y2 - y1, x2 - x1)));
    }else if (compare>240 && compare<=300){
        distance = 335;
        x3 = x1 + distance * cos(double(atan2(y2 - y1, x2 - x1)));
        y3 = y1 + distance * sin(double(atan2(y2 - y1, x2 - x1)));
    }else if (compare>300 && compare<=330){
        distance = 335 + (380-335)*(result-300)/30;
        x3 = x1 + distance * cos(double(atan2(y2 - y1, x2 - x1)));
        y3 = y1 + distance * sin(double(atan2(y2 - y1, x2 - x1)));
    }else if (compare>330 && compare<360){
        distance = 380 + (410-380)*(result-330)/30;
        x3 = x1 + distance * cos(double(atan2(y2 - y1, x2 - x1)));
        y3 = y1 + distance * sin(double(atan2(y2 - y1, x2 - x1)));
    }

    double x = x1 - x3;
    double y = y1 - y3;
    double dis = sqrt(x*x+y*y);
    qDebug() <<"x3:"<<x3<<"y3:"<<y3;
    qDebug() <<"dis:"<<dis;
    return QPoint(x3, y3);
}

void capture::calDistance(const QPoint &pt1, const QPoint &pt2)
{
    double x1 = pt1.x();
    double y1 = pt1.y();
    double x2 = pt2.x();
    double y2 = pt2.y();
    double x = x1 - x2;
    double y = y1 - y2;
    double dis = sqrt(x*x+y*y);
    qDebug() <<"dis:"<<dis;
}



void capture::BfMatch()
{
    if(capMat.empty()){
        return  ;
    }
    Mat img2 = capMat.clone();
    Mat img1 = imread("F:\\CODE_PROJECT\\opencv001\\res\\lolkeg.png");
    Mat gray1,gray2;
    cvtColor(img1,gray1,COLOR_BGR2GRAY);
    cvtColor(img2,gray2,COLOR_BGR2GRAY);
    Ptr<SIFT> sift = SIFT::create();
    std::vector<KeyPoint> kp1,kp2;
    Mat des1,des2;
    sift->detectAndCompute(gray1,Mat(),kp1,des1);
    sift->detectAndCompute(gray2,Mat(),kp2,des2);
    BFMatcher bf = BFMatcher(NORM_L1);
    std::vector<DMatch> matches;
    bf.match(des1,des2,matches);
    Mat img3;
    drawMatches(img1,kp1,img2,kp2,matches,img3);
    imshow("match result",img3);
    waitKey(0);
}

void capture::FlannMatch()
{
    if(capMat.empty()){
        return  ;
    }
    Mat img2 = capMat.clone();
    Mat img1 = imread("F:\\CODE_PROJECT\\opencv001\\res\\lolkeg.png");
    Mat gray1,gray2;
    cvtColor(img1,gray1,COLOR_BGR2GRAY);
    cvtColor(img2,gray2,COLOR_BGR2GRAY);
    Ptr<SIFT> sift = SIFT::create();
    std::vector<KeyPoint> kp1,kp2;
    Mat des1,des2;
    sift->detectAndCompute(gray1,Mat(),kp1,des1);
    sift->detectAndCompute(gray2,Mat(),kp2,des2);

    std::vector<KeyPoint> keypoints_obj,keypoints_scene;
    keypoints_obj = kp1;
    keypoints_scene = kp2;
    Mat descript_obj,descript_scene;
    descript_obj = des1;
    descript_scene = des2;
    FlannBasedMatcher fbmatcher;
    std::vector<DMatch> matches;
    fbmatcher.match(descript_obj, descript_scene, matches); //特征描述子匹配

    //找出最优特征点
    double minDist = 1000;    //初始化最大最小距离
    double maxDist = 0;

    for (int i = 0; i < descript_obj.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist > maxDist)
        {
            maxDist = dist;
        }
        if (dist < minDist)
        {
            minDist = dist;
        }
    }

    qDebug()<<"maxDist:"<<maxDist;
    qDebug()<<"minDist:"<<minDist;

    std::vector<DMatch> goodMatches;
    for (int i = 0; i < descript_obj.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < max(2 * minDist, 0.02)) {
            goodMatches.push_back(matches[i]);
        }
    }

    Mat resultImg;
    drawMatches(img1, keypoints_obj, img2, keypoints_scene, goodMatches, resultImg, Scalar::all(-1),
                Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
                );

    imshow("FlannBasedMatcher demo", resultImg);
    waitKey(0);

}



void capture::SiftBf()
{
    if(capMat.empty()){
        return  ;
    }
    Mat img2 = capMat.clone();
    Mat img1 = imread("F:\\CODE_PROJECT\\opencv001\\res\\lolkeg.png");
    //灰度
    Mat gray1,gray2;
    cvtColor(img1,gray1,COLOR_BGR2GRAY);
    cvtColor(img2,gray2,COLOR_BGR2GRAY);
    //SIFT特征提取
    Ptr<SIFT> sift = SIFT::create();
    std::vector<KeyPoint> kp1,kp2;
    Mat des1,des2;
    sift->detectAndCompute(gray1,Mat(),kp1,des1);
    sift->detectAndCompute(gray2,Mat(),kp2,des2);
    std::vector<KeyPoint> keypoints_obj,keypoints_scene;
    keypoints_obj = kp1;
    keypoints_scene = kp2;
    Mat descript_obj,descript_scene;
    descript_obj = des1;
    descript_scene = des2;
    //暴力特征匹配
    BFMatcher bf = BFMatcher(NORM_L1);
    std::vector<DMatch> matches;
    bf.match(descript_obj, descript_scene, matches); //特征描述子匹配


    double minDist = 1000;    //初始化最大最小距离
    double maxDist = 0;
    for (int i = 0; i < descript_obj.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist > maxDist)
        {
            maxDist = dist;
        }
        if (dist < minDist)
        {
            minDist = dist;
        }
    }
    qDebug()<<"maxDist:"<<maxDist;
    qDebug()<<"minDist:"<<minDist;

    std::vector<DMatch> goodMatches;
    for (int i = 0; i < descript_obj.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < 100) {
            goodMatches.push_back(matches[i]);
        }
    }

    if(goodMatches.size()<4){
        qDebug()<<"match fault";
        return;
    }

    std::vector<Point2f>point1, point2;
    for (int i = 0; i < goodMatches.size(); i++)
    {
        point1.push_back(kp1[goodMatches[i].queryIdx].pt);
        point2.push_back(kp2[goodMatches[i].trainIdx].pt);
    }

    Mat H = findHomography(point1, point2, RANSAC,5);
    std::vector<Point2f>cornerPoints1(4);
    std::vector<Point2f>cornerPoints2(4);
    cornerPoints1[0] = Point(0, 0);
    cornerPoints1[1] = Point(img1.cols, 0);
    cornerPoints1[2] = Point(img1.cols, img1.rows);
    cornerPoints1[3] = Point(0,img1.rows);
    perspectiveTransform(cornerPoints1, cornerPoints2, H);

    //在原图上绘制出变换后的目标轮廓
    line(img2, cornerPoints2[0], cornerPoints2[1], Scalar(0,255,255), 4, 8, 0);
    line(img2, cornerPoints2[1], cornerPoints2[2], Scalar(0,255,255), 4, 8, 0);
    line(img2, cornerPoints2[2], cornerPoints2[3], Scalar(0,255,255), 4, 8, 0);
    line(img2, cornerPoints2[3], cornerPoints2[0], Scalar(0,255,255), 4, 8, 0);

    imshow("result", img2);
    waitKey(0);
}



void capture::SiftFlann()
{
    QTime time;
    time.start();
    if(capMat.empty()){
        return  ;
    }
    Mat img2 = capMat.clone();
    Mat img1 = imread("F:\\CODE_PROJECT\\opencv002\\res\\lolkeg1.png");
    Mat gray1,gray2;
    cvtColor(img1,gray1,COLOR_BGR2GRAY);
    cvtColor(img2,gray2,COLOR_BGR2GRAY);
    Ptr<SIFT> sift = SIFT::create(300);
    std::vector<KeyPoint> kp1,kp2;
    Mat des1,des2;
    sift->detectAndCompute(gray1,Mat(),kp1,des1);
    qDebug()<<"time0= "<< time.elapsed()/1000.0<<"s";
    sift->detectAndCompute(gray2,Mat(),kp2,des2);
    qDebug()<<"time1= "<< time.elapsed()/1000.0<<"s";
    qDebug()<<"kp1= "<<kp1.size()<<" kp2= "<<kp2.size();
    std::vector<KeyPoint> keypoints_obj,keypoints_scene;
    keypoints_obj = kp1;
    keypoints_scene = kp2;
    Mat descript_obj,descript_scene;
    descript_obj = des1;
    descript_scene = des2;
    FlannBasedMatcher fbmatcher;
    std::vector<DMatch> matches;
    fbmatcher.match(descript_obj, descript_scene, matches); //特征描述子匹配
    qDebug()<<"time2= "<< time.elapsed()/1000.0<<"s";
    //找出最优特征点
    double minDist = 1000;    //初始化最大最小距离
    double maxDist = 0;

    for (int i = 0; i < descript_obj.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist > maxDist)
        {
            maxDist = dist;
        }
        if (dist < minDist)
        {
            minDist = dist;
        }
    }
    qDebug()<<"maxDist:"<<maxDist;
    qDebug()<<"minDist:"<<minDist;

    std::vector<DMatch> goodMatches;
    for (int i = 0; i < descript_obj.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < max((minDist*2),0.02)) {
            goodMatches.push_back(matches[i]);
        }
    }
    qDebug()<<"goodMatches::"<<goodMatches.size();
    if(goodMatches.size()<4){
        qDebug()<<"match fault";
        return;
    }

    std::vector<Point2f>point1, point2;
    for (int i = 0; i < goodMatches.size(); i++)
    {
        point1.push_back(kp1[goodMatches[i].queryIdx].pt);
        point2.push_back(kp2[goodMatches[i].trainIdx].pt);
    }

    Mat H = findHomography(point1, point2, RANSAC,5);
    std::vector<Point2f>cornerPoints1(4);
    std::vector<Point2f>cornerPoints2(4);
    cornerPoints1[0] = Point(0, 0);
    cornerPoints1[1] = Point(img1.cols, 0);
    cornerPoints1[2] = Point(img1.cols, img1.rows);
    cornerPoints1[3] = Point(0,img1.rows);
    perspectiveTransform(cornerPoints1, cornerPoints2, H);
    qDebug()<<"total time= "<< time.elapsed()/1000.0<<"s";
    //在原图上绘制出变换后的目标轮廓
    line(img2, cornerPoints2[0], cornerPoints2[1], Scalar(0,255,255), 4, 8, 0);
    line(img2, cornerPoints2[1], cornerPoints2[2], Scalar(0,255,255), 4, 8, 0);
    line(img2, cornerPoints2[2], cornerPoints2[3], Scalar(0,255,255), 4, 8, 0);
    line(img2, cornerPoints2[3], cornerPoints2[0], Scalar(0,255,255), 4, 8, 0);

    imshow("result", img2);
    waitKey(0);
}

void capture::surfFlann()
{
    QTime time;
    time.start();
    if(capMat.empty()){
        return  ;
    }
    Mat img2 = capMat.clone();
    Mat img1 = imread("F:\\CODE_PROJECT\\opencv002\\res\\lolkeg1.png");
    if(img1.empty()){
        return  ;
    }
    Mat gray1,gray2,gaussian1,gaussian2;
    cvtColor(img1,gray1,COLOR_RGB2GRAY);
    GaussianBlur(img1, gaussian1, Size(3, 3), 0);
    cvtColor(img2,gray2,COLOR_RGB2GRAY);
    GaussianBlur(img2, gaussian2, Size(3, 3), 0);
    Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(1000);
    std::vector<KeyPoint> kp1,kp2;
    Mat des1,des2;
    surf->detectAndCompute(gaussian1,Mat(),kp1,des1);
    qDebug()<<"time0= "<< time.elapsed()/1000.0<<"s";
    surf->detectAndCompute(gaussian2,Mat(),kp2,des2);
    qDebug()<<"time1= "<< time.elapsed()/1000.0<<"s";
    qDebug()<<"kp1= "<<kp1.size()<<" kp2= "<<kp2.size();
    std::vector<KeyPoint> keypoints_obj,keypoints_scene;
    keypoints_obj = kp1;
    keypoints_scene = kp2;
    Mat descript_obj,descript_scene;
    descript_obj = des1;
    descript_scene = des2;
    FlannBasedMatcher fbmatcher;
    std::vector<DMatch> matches;
    fbmatcher.match(descript_obj, descript_scene, matches); //特征描述子匹配
    qDebug()<<"time2= "<< time.elapsed()/1000.0<<"s";
    //找出最优特征点
    double minDist = 1000;    //初始化最大最小距离
    double maxDist = 0;

    for (int i = 0; i < descript_obj.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist > maxDist)
        {
            maxDist = dist;
        }
        if (dist < minDist)
        {
            minDist = dist;
        }
    }
    qDebug()<<"maxDist:"<<maxDist;
    qDebug()<<"minDist:"<<minDist;

    std::vector<DMatch> goodMatches;
    for (int i = 0; i < descript_obj.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < max((minDist*2),0.02)) {
            goodMatches.push_back(matches[i]);
        }
    }
    qDebug()<<"goodMatches::"<<goodMatches.size();
    if(goodMatches.size()<4){
        qDebug()<<"match fault";
        return;
    }

    std::vector<Point2f>point1, point2;
    for (int i = 0; i < goodMatches.size(); i++)
    {
        point1.push_back(kp1[goodMatches[i].queryIdx].pt);
        point2.push_back(kp2[goodMatches[i].trainIdx].pt);
    }

    Mat H = findHomography(point1, point2, RANSAC,5);
    std::vector<Point2f>cornerPoints1(4);
    std::vector<Point2f>cornerPoints2(4);
    cornerPoints1[0] = Point(0, 0);
    cornerPoints1[1] = Point(img1.cols, 0);
    cornerPoints1[2] = Point(img1.cols, img1.rows);
    cornerPoints1[3] = Point(0,img1.rows);
    perspectiveTransform(cornerPoints1, cornerPoints2, H);
    qDebug()<<"total time= "<< time.elapsed()/1000.0<<"s";
    //在原图上绘制出变换后的目标轮廓
    line(img2, cornerPoints2[0], cornerPoints2[1], Scalar(0,255,255), 4, 8, 0);
    line(img2, cornerPoints2[1], cornerPoints2[2], Scalar(0,255,255), 4, 8, 0);
    line(img2, cornerPoints2[2], cornerPoints2[3], Scalar(0,255,255), 4, 8, 0);
    line(img2, cornerPoints2[3], cornerPoints2[0], Scalar(0,255,255), 4, 8, 0);

    imshow("result", img2);
    waitKey(0);
}

void capture::orbBf()
{
    if(capMat.empty()){
        return  ;
    }
    Mat img2 = capMat.clone();
    Mat img1 = imread("F:\\CODE_PROJECT\\opencv001\\res\\lolkeg.png");
    Mat gray1,gray2;
    cvtColor(img1,gray1,COLOR_BGR2GRAY);
    cvtColor(img2,gray2,COLOR_BGR2GRAY);
    Ptr<ORB> orb = ORB::create();
    std::vector<KeyPoint> kp1,kp2;
    Mat des1,des2;
    orb->detectAndCompute(gray1,Mat(),kp1,des1);
    orb->detectAndCompute(gray2,Mat(),kp2,des2);

    std::vector<KeyPoint> keypoints_obj,keypoints_scene;
    keypoints_obj = kp1;
    keypoints_scene = kp2;
    Mat descript_obj,descript_scene;
    descript_obj = des1;
    descript_scene = des2;
    //暴力特征匹配
    BFMatcher bf = BFMatcher(NORM_HAMMING);
    std::vector<DMatch> matches;
    bf.match(descript_obj, descript_scene, matches); //特征描述子匹配

    //找出最优特征点
    double minDist = 1000;    //初始化最大最小距离
    double maxDist = 0;

    for (int i = 0; i < descript_obj.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist > maxDist)
        {
            maxDist = dist;
        }
        if (dist < minDist)
        {
            minDist = dist;
        }
    }
    qDebug()<<"maxDist:"<<maxDist;
    qDebug()<<"minDist:"<<minDist;

    std::vector<DMatch> goodMatches;
    for (int i = 0; i < descript_obj.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < max(2 * minDist, 0.02)) {
            goodMatches.push_back(matches[i]);
        }
    }

    if(goodMatches.size()<4){
        qDebug()<<"match fault";
        return;
    }

    std::vector<Point2f>point1, point2;
    for (int i = 0; i < goodMatches.size(); i++)
    {
        point1.push_back(kp1[goodMatches[i].queryIdx].pt);
        point2.push_back(kp2[goodMatches[i].trainIdx].pt);
    }

    Mat H = findHomography(point1, point2, RANSAC,5);
    std::vector<Point2f>cornerPoints1(4);
    std::vector<Point2f>cornerPoints2(4);
    cornerPoints1[0] = Point(0, 0);
    cornerPoints1[1] = Point(img1.cols, 0);
    cornerPoints1[2] = Point(img1.cols, img1.rows);
    cornerPoints1[3] = Point(0,img1.rows);
    perspectiveTransform(cornerPoints1, cornerPoints2, H);

    //在原图上绘制出变换后的目标轮廓
    line(img2, cornerPoints2[0], cornerPoints2[1], Scalar(0,255,255), 4, 8, 0);
    line(img2, cornerPoints2[1], cornerPoints2[2], Scalar(0,255,255), 4, 8, 0);
    line(img2, cornerPoints2[2], cornerPoints2[3], Scalar(0,255,255), 4, 8, 0);
    line(img2, cornerPoints2[3], cornerPoints2[0], Scalar(0,255,255), 4, 8, 0);

    imshow("result", img2);
    waitKey(0);
}

void capture::OrbFlann()
{
    if(capMat.empty()){
        return  ;
    }
    Mat img2 = capMat.clone();
    Mat img1 = imread("F:\\CODE_PROJECT\\opencv001\\res\\lolkeg.png");
    Mat gray1,gray2;
    cvtColor(img1,gray1,COLOR_BGR2GRAY);
    cvtColor(img2,gray2,COLOR_BGR2GRAY);
    Ptr<ORB> orb = ORB::create();
    std::vector<KeyPoint> kp1,kp2;
    Mat des1,des2;
    orb->detectAndCompute(gray1,Mat(),kp1,des1);
    orb->detectAndCompute(gray2,Mat(),kp2,des2);

    std::vector<KeyPoint> keypoints_obj,keypoints_scene;
    keypoints_obj = kp1;
    keypoints_scene = kp2;
    Mat descript_obj,descript_scene;
    descript_obj = des1;
    descript_scene = des2;

    if(descript_obj.type() != CV_32F && descript_obj.type() != CV_32F ){
        descript_obj.convertTo(descript_obj,CV_32F);
        descript_scene.convertTo(descript_scene,CV_32F);
    }

    std::vector<DMatch> matches;
    FlannBasedMatcher fbmatcher;
    fbmatcher.match(descript_obj,descript_scene,matches); //特征描述子匹配

    //找出最优特征点
    double minDist = 1000;    //初始化最大最小距离
    double maxDist = 0;

    for (int i = 0; i < descript_obj.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist > maxDist)
        {
            maxDist = dist;
        }
        if (dist < minDist)
        {
            minDist = dist;
        }
    }

    qDebug()<<"maxDist:"<<maxDist;
    qDebug()<<"minDist:"<<minDist;

    std::vector<DMatch> goodMatches;
    for (int i = 0; i < descript_obj.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < max(2 * minDist, 0.02)) {
            goodMatches.push_back(matches[i]);
        }
    }

    if(goodMatches.size()<4){
        qDebug()<<"match fault";
        return;
    }

    std::vector<Point2f>point1, point2;
    for (int i = 0; i < goodMatches.size(); i++)
    {
        point1.push_back(kp1[goodMatches[i].queryIdx].pt);
        point2.push_back(kp2[goodMatches[i].trainIdx].pt);
    }

    Mat H = findHomography(point1, point2, RANSAC);
    std::vector<Point2f>cornerPoints1(4);
    std::vector<Point2f>cornerPoints2(4);
    cornerPoints1[0] = Point(0, 0);
    cornerPoints1[1] = Point(img1.cols, 0);
    cornerPoints1[2] = Point(img1.cols, img1.rows);
    cornerPoints1[3] = Point(0,img1.rows);
    perspectiveTransform(cornerPoints1, cornerPoints2, H);

    //在原图上绘制出变换后的目标轮廓
    line(img2, cornerPoints2[0], cornerPoints2[1], Scalar(0,255,255), 4, 8, 0);
    line(img2, cornerPoints2[1], cornerPoints2[2], Scalar(0,255,255), 4, 8, 0);
    line(img2, cornerPoints2[2], cornerPoints2[3], Scalar(0,255,255), 4, 8, 0);
    line(img2, cornerPoints2[3], cornerPoints2[0], Scalar(0,255,255), 4, 8, 0);

    imshow("result", img2);
    waitKey(0);
}

void capture::preWork()
{
    qDebug() <<"x:"<<QCursor().pos().x()<<"y:"<<QCursor().pos().y();
    if(capMat.empty()){
        return  ;
    }
    QTime time;
    time.start();
    cv::Mat src = capMat.clone();
    cv::Mat sample = imread("F:\\CODE_PROJECT\\opencv002\\res\\lolkeg1.png");
    if(sample.empty()){
        qDebug() <<"sample is empty";
        return  ;
    }
    int col= sample.cols;
    int row= sample.rows;
    qDebug() <<"sample.cols:"<<col<<" sample.rows:"<<row;
    qDebug()<<"src.type:"<<src.type()<<"sample.type:"<<sample.type();
    qDebug()<<"time0= "<< time.elapsed()/1000.0<<"s";
    namedWindow("threshold result", WINDOW_AUTOSIZE);
    createTrackbar("Threshold", "threshold result", &thresh, 255, thresh_Demo);
    thresh_Demo(0, 0);
    waitKey(0);

    //    imshow("THRESH", sample_thresh);
    //    waitKey(0);
    //    imshow("src1", src_down);
    //    waitKey(0);
    //    imshow("sample1", sample_down);
    //    waitKey(0);
}

void capture::colorDetc()
{
    QTime time;
    time.start();
    cv::Mat src = capMat.clone();
    cv::Mat sample = imread("F:\\CODE_PROJECT\\opencv002\\res\\lolkeg1.png");
    if(sample.empty()){
        qDebug() <<"sample is empty";
        return  ;
    }
    Mat src_hsv;
    cvtColor(src,src_hsv,COLOR_BGR2HSV);
    imshow("hsv", src_hsv);
    waitKey(0);
    Mat result;
    inRange(src_hsv,Scalar(0,242,51),Scalar(4,255,64),result);
    imshow("result", result);
    waitKey(0);
    //    Mat final;
    //    bitwise_and()

}



void capture::Gp()
{
    qDebug() <<"GpKeg start";
    while(1)
    {
        short flag2 = GetKeyState(Qt::Key_2);
        //        qDebug()<<flag;
        if(flag2<0){
            GpKeg();
        }

        //        short flag1 = GetKeyState(Qt::Key_1);
        //        if(flag1<0){
        //            GpEnemy();
        //        }

        //        short flag3 = GetKeyState(Qt::Key_3);
        //        if(flag3<0){
        //            GpTest();
        //        }

    }
}

void capture::GpEnemy()
{
    QTime time;
    time.start();
    HWND hwnd = GetDesktopWindow();
    Mat screen = captureScreenMat(hwnd);
    qDebug()<<"screenshot time= "<< time.elapsed()/1000.0<<"s";
    if(screen.empty()){
        return  ;
    }
    cv::Mat src = screen.clone();
    cv::Mat sample = imread("F:\\CODE_PROJECT\\opencv002\\res\\en2.png");
    cv::Mat mask = imread("F:\\CODE_PROJECT\\opencv002\\res\\en1.png");
    qDebug() <<"screen type"<<src.type();
    cvtColor(src,src,COLOR_RGBA2RGB);
    qDebug() <<"screen type"<<src.type();
    qDebug() <<"temp type"<<sample.type();
    if(sample.empty()){
        qDebug() <<"sample is empty";
        return  ;
    }
    //预处理
    Mat src_down,sample_down,mask_down;
    pyrDown(src,src_down);
    pyrDown(sample,sample_down);
    pyrDown(mask,mask_down);
    //模板匹配
    Mat result;
    matchTemplate(src_down, sample_down, result, TM_CCORR_NORMED,mask_down);
    qDebug()<<"time2= "<< time.elapsed()/1000.0<<"s";

    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    qDebug()<<"maxVal"<<maxVal;
    qDebug()<<"maxLoc("<<maxLoc.x << ","<<maxLoc.y<<")";
    qDebug()<<"time3= "<< time.elapsed()/1000.0<<"s";
    if(maxVal<0.94){
        qDebug()<<"match failed";
        return;
    }
    else {
        qDebug()<<"total time= "<< time.elapsed()/1000.0<<"s";
        qDebug()<<"resultPoint("<<maxLoc.x*2 << ","<<maxLoc.y*2<<")";
        DD_mov(maxLoc.x*2+75,maxLoc.y*2+150);
        Sleep(20);
        DD_key(301,1);
        Sleep(20);
        DD_key(301,2);
    }
}


void capture::GpTest()
{
    //    qDebug() <<"x:"<<QCursor().pos().x()<<"y:"<<QCursor().pos().y();
    //    Point m_cursor;
    //    m_cursor.x = QCursor().pos().x();
    //    m_cursor.y = QCursor().pos().y();
    //    QTime time;
    //    time.start();
    //    cv::Mat src = capMat.clone();
    //    if(src.empty()){
    //        return  ;
    //    }

    //    double x1 = 960;
    //    double y1 = 540;
    //    double x2 = 0.0;
    //    double y2 = 0.0;
    //    double x3 = 0.0, y3 = 0.0;

    //    int angle = 330;//逆时针度数
    //    int trans_angle = 360 - angle;
    //    x3 = x1 + 100 * cos(trans_angle*M_PI/180);
    //    y3 = y1 + 100 * sin(trans_angle*M_PI/180);

    //    double x = x1 - x3;
    //    double y = y1 - y3;
    //    double dis = sqrt(x*x+y*y);
    //    qDebug() <<"x3:"<<x3<<"y3:"<<y3;
    //    qDebug() <<"dis:"<<dis;

    //    double result = atan2(y3 - y1, x3 - x1)*180/M_PI;//-180~180度 逆时针度数
    //    result = -result;
    //    if(result<0)result = result + 360;
    //    qDebug()<<"result angle"<<result;

    //    Mat draw = src.clone();
    //    arrowedLine(draw, Point(480, 540), Point(1440, 540), Scalar(0, 0, 255), 2, 8, 0);
    //    arrowedLine(draw, Point(960, 810), Point(960, 270), Scalar(0, 0, 255), 2, 8, 0);
    //    char text[10];
    //    sprintf_s(text, "%d",angle);
    //    putText(draw, text, Point(x3, y3), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
    //    arrowedLine(draw, Point(x1, y1), Point(x3, y3), Scalar(0, 0, 255), 2, 8, 0);
    //    qDebug()<<"time= "<< time.elapsed()/1000.0<<"s";
    //    imshow("draw", draw);
    //    waitKey(0);



    qDebug() <<"x:"<<QCursor().pos().x()<<"y:"<<QCursor().pos().y();
    Point m_cursor;
    m_cursor.x = QCursor().pos().x();
    m_cursor.y = QCursor().pos().y();
    QTime time;
    time.start();
    HWND hwnd = GetDesktopWindow();
    Mat screen = captureScreenMat(hwnd);
    qDebug()<<"screenshot time= "<< time.elapsed()/1000.0<<"s";
    if(screen.empty()){
        return  ;
    }
    cv::Mat src = screen.clone();
    cv::Mat sample = imread("F:\\CODE_PROJECT\\opencv002\\res\\lolkeg1.png");
    qDebug() <<"screen type"<<src.type();
    cvtColor(src,src,COLOR_RGBA2RGB);
    qDebug() <<"screen type"<<src.type();
    qDebug() <<"temp type"<<sample.type();
    if(sample.empty()){
        qDebug() <<"sample is empty";
        return  ;
    }
    //预处理
    Mat src_down,sample_down;
    pyrDown(src,src_down);
    pyrDown(sample,sample_down);
    //模板匹配
    Mat result;
    matchTemplate(src_down, sample_down, result, TM_CCORR_NORMED);
    qDebug()<<"time2= "<< time.elapsed()/1000.0<<"s";

    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    qDebug()<<"maxVal"<<maxVal;
    qDebug()<<"maxLoc("<<maxLoc.x << ","<<maxLoc.y<<")";
    qDebug()<<"time3= "<< time.elapsed()/1000.0<<"s";
    if(maxVal<0.9){
        qDebug()<<"temp match failed";
        return;
    }

    //多目标匹配
    QList<QPoint> listResult;
    for (int i = 0; i < result.rows; i++)
    {
        for (int j = 0; j < result.cols; j++)
        {
            double val = result.at<float>(i, j);//灰度值

            //3*3邻域非极大值抑制
            if (val > 0.897)
            {
                if(i!=0 && j!=0 && i!=(result.rows-1) && j!=(result.cols-1)){
                    //当前像素的灰度值大于阈值且该像素是其3*3邻域最值时，判定其为目标
                    if (result.at<float>(i - 1, j - 1) < val &&
                            result.at<float>(i - 1, j) < val &&
                            result.at<float>(i - 1, j + 1) < val &&
                            result.at<float>(i, j - 1) < val &&
                            result.at<float>(i, j + 1) < val &&
                            result.at<float>(i + 1, j - 1) < val &&
                            result.at<float>(i + 1, j) < val &&
                            result.at<float>(i + 1, j + 1) < val)
                    {
                        qDebug()<<"result="<<val;
                        qDebug()<<"(x="<<j<<",y="<<i<<")";
                        listResult.append(QPoint(j*2, i*2));
                    }
                }
            }
        }
    }
    qDebug()<<"time3= "<< time.elapsed()/1000.0<<"s";
    //特征匹配
    qDebug()<<"temple result= "<< listResult.size();
    if(listResult.size()>0){
        int best_index = 0;
        double temp_min = 1000;
        for(int i=0;i<listResult.size();i++)
        {
            Mat imgRst = src(Rect(listResult.at(i).x(), listResult.at(i).y(), sample.cols, sample.rows));
            qDebug()<<listResult.at(i).x()<<listResult.at(i).y()<<sample.cols<<sample.rows;
            std::vector<KeyPoint> kp;
            Mat des;
            cvtColor(imgRst,imgRst,COLOR_RGB2GRAY);
            surf_detector->detectAndCompute(imgRst,Mat(),kp,des);
            qDebug()<<"Total keyponits: "<<kp.size();
            Mat descript_obj,descript_scene;
            descript_obj = des;
            descript_scene = keg_des;
            FlannBasedMatcher fbmatcher;
            std::vector<DMatch> matches;
            fbmatcher.match(descript_obj, descript_scene, matches); //特征描述子匹配
            qDebug()<<"time= "<< time.elapsed()/1000.0<<"s";
            //找出最优特征点
            double minDist = 1000;    //初始化最大最小距离
            double maxDist = 0;

            for (int i = 0; i < descript_obj.rows; i++)
            {
                double dist = matches[i].distance;
                if (dist > maxDist)
                {
                    maxDist = dist;
                }
                if (dist < minDist)
                {
                    minDist = dist;
                }
            }
            qDebug()<<"maxDist:"<<maxDist;
            qDebug()<<"minDist:"<<minDist;

            if(minDist>0.2){
                qDebug()<<"match fault"<<i;
            }
            else {
                //特征识别结果保存
                if(minDist<temp_min)
                {
                    temp_min = minDist;
                    best_index = i;
                }
            }
        }
        if(temp_min<0.2)
        {
            qDebug()<<"temp_min"<<temp_min<<" best_index"<<best_index;
            Point matchF;
            matchF.x = listResult.at(best_index).x()+48;
            matchF.y = listResult.at(best_index).y()+48;
            qDebug()<<"resultPoint("<<matchF.x << ","<<matchF.y<<")";

            //            QPoint calResult = calSecondKeg(QPoint(matchF.x,matchF.y),QPoint(m_cursor.x,m_cursor.y));

            double x3 = 0.0, y3 = 0.0;
            int angle = 240;//逆时针度数
            int distance = 340;
            int trans_angle = 360 - angle;
            x3 = matchF.x + distance * cos(trans_angle*M_PI/180);
            y3 = matchF.y + distance * sin(trans_angle*M_PI/180);


            qDebug()<<"maxVal"<<maxVal;
            qDebug()<<"maxLoc("<<maxLoc.x << ","<<maxLoc.y<<")";
            qDebug()<<"temp result= "<< listResult.size();
            qDebug()<<"total time= "<< time.elapsed()/1000.0<<"s";

            //            DD_mov(calResult.x(),calResult.y());
            DD_mov(x3,y3);
            Sleep(20);
            DD_key(303,1);
            Sleep(20);
            DD_key(303,2);
            Sleep(300);
            DD_mov(matchF.x,matchF.y);
            Sleep(20);
            DD_key(301,1);
            Sleep(20);
            DD_key(301,2);
        }
    }
}


void capture::GpKeg()
{
    qDebug() <<"x:"<<QCursor().pos().x()<<"y:"<<QCursor().pos().y();
    Point m_cursor;
    m_cursor.x = QCursor().pos().x();
    m_cursor.y = QCursor().pos().y();
    QTime time;
    time.start();
    HWND hwnd = GetDesktopWindow();
    Mat screen = captureScreenMat(hwnd);
    qDebug()<<"screenshot time= "<< time.elapsed()/1000.0<<"s";
    if(screen.empty()){
        return  ;
    }
    cv::Mat src = screen.clone();
    cv::Mat sample = imread("F:\\CODE_PROJECT\\opencv002\\res\\lolkeg1.png");
    qDebug() <<"screen type"<<src.type();
    cvtColor(src,src,COLOR_RGBA2RGB);
    qDebug() <<"screen type"<<src.type();
    qDebug() <<"temp type"<<sample.type();
    if(sample.empty()){
        qDebug() <<"sample is empty";
        return  ;
    }
    //预处理
    Mat src_down,sample_down;
    pyrDown(src,src_down);
    pyrDown(sample,sample_down);
    //    Mat src_gray,sample_gray;
    //    cvtColor(src_down,src_gray,COLOR_RGB2GRAY);
    //    cvtColor(sample_down,sample_gray,COLOR_RGB2GRAY);
    //    Mat src_thresh,sample_thresh;
    //    cv::threshold(src_down,src_thresh,85,255,THRESH_BINARY);
    //    cv::threshold(sample_down,sample_thresh,85,255,THRESH_BINARY);
    //模板匹配
    Mat result;
    matchTemplate(src_down, sample_down, result, TM_CCORR_NORMED);
    qDebug()<<"time2= "<< time.elapsed()/1000.0<<"s";

    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    qDebug()<<"maxVal"<<maxVal;
    qDebug()<<"maxLoc("<<maxLoc.x << ","<<maxLoc.y<<")";
    qDebug()<<"time3= "<< time.elapsed()/1000.0<<"s";
    if(maxVal<0.9){
        qDebug()<<"temp match failed";
        return;
    }

    //多目标匹配
    QList<QPoint> listResult;
    for (int i = 0; i < result.rows; i++)
    {
        for (int j = 0; j < result.cols; j++)
        {
            double val = result.at<float>(i, j);//灰度值

            //3*3邻域非极大值抑制
            if (val > 0.898)
            {
                if(i!=0 && j!=0 && i!=(result.rows-1) && j!=(result.cols-1)){
                    //当前像素的灰度值大于阈值且该像素是其3*3邻域最值时，判定其为目标
                    if (result.at<float>(i - 1, j - 1) < val &&
                            result.at<float>(i - 1, j) < val &&
                            result.at<float>(i - 1, j + 1) < val &&
                            result.at<float>(i, j - 1) < val &&
                            result.at<float>(i, j + 1) < val &&
                            result.at<float>(i + 1, j - 1) < val &&
                            result.at<float>(i + 1, j) < val &&
                            result.at<float>(i + 1, j + 1) < val)
                    {
                        qDebug()<<"result="<<val;
                        qDebug()<<"(x="<<j<<",y="<<i<<")";
                        listResult.append(QPoint(j*2, i*2));
                    }
                }
            }
        }
    }
    qDebug()<<"time3= "<< time.elapsed()/1000.0<<"s";
    //特征匹配
    qDebug()<<"temple result= "<< listResult.size();
    if(listResult.size()>0){
        int best_index = 0;
        double temp_min = 1000;
        for(int i=0;i<listResult.size();i++)
        {
            Mat imgRst = src(Rect(listResult.at(i).x(), listResult.at(i).y(), sample.cols, sample.rows));
            qDebug()<<listResult.at(i).x()<<listResult.at(i).y()<<sample.cols<<sample.rows;
            std::vector<KeyPoint> kp;
            Mat des;
            cvtColor(imgRst,imgRst,COLOR_RGB2GRAY);
            surf_detector->detectAndCompute(imgRst,Mat(),kp,des);
            qDebug()<<"Total keyponits: "<<kp.size();
            Mat descript_obj,descript_scene;
            descript_obj = des;
            descript_scene = keg_des;
            FlannBasedMatcher fbmatcher;
            std::vector<DMatch> matches;
            fbmatcher.match(descript_obj, descript_scene, matches); //特征描述子匹配
            qDebug()<<"time= "<< time.elapsed()/1000.0<<"s";
            //找出最优特征点
            double minDist = 1000;    //初始化最大最小距离
            double maxDist = 0;

            for (int i = 0; i < descript_obj.rows; i++)
            {
                double dist = matches[i].distance;
                if (dist > maxDist)
                {
                    maxDist = dist;
                }
                if (dist < minDist)
                {
                    minDist = dist;
                }
            }
            qDebug()<<"maxDist:"<<maxDist;
            qDebug()<<"minDist:"<<minDist;

            if(minDist>0.2){
                qDebug()<<"match fault"<<i;
            }
            else {
                //特征识别结果保存
                if(minDist<temp_min)
                {
                    temp_min = minDist;
                    best_index = i;
                }
            }
        }
        if(temp_min<0.2)
        {
            qDebug()<<"temp_min"<<temp_min<<" best_index"<<best_index;
            Point matchF;
            matchF.x = listResult.at(best_index).x()+48;
            matchF.y = listResult.at(best_index).y()+48;
            qDebug()<<"resultPoint("<<matchF.x << ","<<matchF.y<<")";
            QPoint calResult = calSecondKeg1(QPoint(matchF.x,matchF.y),QPoint(m_cursor.x,m_cursor.y));
            qDebug()<<"maxVal"<<maxVal;
            qDebug()<<"maxLoc("<<maxLoc.x << ","<<maxLoc.y<<")";
            qDebug()<<"temp result= "<< listResult.size();
            qDebug()<<"total time= "<< time.elapsed()/1000.0<<"s";

            DD_mov(calResult.x(),calResult.y());
            Sleep(20);
            DD_key(303,1);
            Sleep(20);
            DD_key(303,2);
            Sleep(300);
            DD_mov(matchF.x,matchF.y);
            Sleep(20);
            DD_key(301,1);
            Sleep(20);
            DD_key(301,2);
        }
    }

    //    Mat imgRst = src(Rect(maxLoc.x*2, maxLoc.y*2, sample.cols, sample.rows));
    //    qDebug()<<maxLoc.x<<maxLoc.y<<sample.cols<<sample.rows;
    //    //    imshow(" match ", imgRst);
    //    //    waitKey(0);
    //    std::vector<KeyPoint> kp;
    //    Mat des;
    //    cvtColor(imgRst,imgRst,COLOR_RGB2GRAY);
    //    surf_detector->detectAndCompute(imgRst,Mat(),kp,des);
    //    qDebug()<<"imgRst keyponits: "<<kp.size();
    //    Mat descript_obj,descript_scene;
    //    descript_obj = des;
    //    descript_scene = keg_des;
    //    FlannBasedMatcher fbmatcher;
    //    std::vector<DMatch> matches;
    //    fbmatcher.match(descript_obj, descript_scene, matches); //特征描述子匹配
    //    qDebug()<<"time2= "<< time.elapsed()/1000.0<<"s";
    //    //找出最优特征点
    //    double minDist = 1000;    //初始化最大最小距离
    //    double maxDist = 0;

    //    for (int i = 0; i < descript_obj.rows; i++)
    //    {
    //        double dist = matches[i].distance;
    //        if (dist > maxDist)
    //        {
    //            maxDist = dist;
    //        }
    //        if (dist < minDist)
    //        {
    //            minDist = dist;
    //        }
    //    }
    //    qDebug()<<"maxDist:"<<maxDist;
    //    qDebug()<<"minDist:"<<minDist;

    //    std::vector<DMatch> goodMatches;
    //    for (int i = 0; i < descript_obj.rows; i++)
    //    {
    //        double dist = matches[i].distance;
    //        if (dist < 0.18) {
    //            goodMatches.push_back(matches[i]);
    //        }
    //    }
    //    qDebug()<<"goodMatches::"<<goodMatches.size();
    //    if(minDist>0.2){
    //        qDebug()<<"surf match fault";
    //    }
    //    else {
    //        Point matchF;
    //        matchF.x = maxLoc.x*2+48;
    //        matchF.y = maxLoc.y*2+48;
    //        qDebug()<<"resultPoint("<<matchF.x << ","<<matchF.y<<")";
    //        QPoint calResult = calSecondKeg(QPoint(matchF.x,matchF.y),QPoint(m_cursor.x,m_cursor.y));
    //        qDebug()<<"time4= "<< time.elapsed()/1000.0<<"s";

    //        DD_mov(calResult.x(),calResult.y());
    //        Sleep(20);
    //        DD_key(303,1);
    //        Sleep(20);
    //        DD_key(303,2);
    //        Sleep(300);
    //        DD_mov(matchF.x,matchF.y);
    //        Sleep(20);
    //        DD_key(301,1);
    //        Sleep(20);
    //        DD_key(301,2);
    //    }

    //        // 框选结果
    //        cv::Mat draw = src.clone();
    //        rectangle(draw, matchF, Point(matchF.x + sample.cols, matchF.y + sample.rows), Scalar(0, 0, 255), 2, 8, 0);
    //        qDebug()<<"total_time= "<< time.elapsed()/1000.0<<"s";
    //        imshow("draw", draw);
    //        waitKey(0);

}



void capture::GpDetcSiftFlann()
{
    QTime time;
    time.start();
    HWND hwnd = GetDesktopWindow();
    Mat screen = captureScreenMat(hwnd);
    qDebug()<<"screenshot time= "<< time.elapsed()/1000.0<<"s";

    if(screen.empty()){
        return  ;
    }
    cv::Mat src = screen.clone();
    cv::Mat temp = imread("F:\\CODE_PROJECT\\opencv001\\res\\lolkeg1.png");
    if(temp.empty()){
        qDebug() <<"sample is empty";
        return  ;
    }


    Mat img2 = src;
    Mat img1 = temp;
    Mat gray1,gray2;
    cvtColor(img1,gray1,COLOR_BGR2GRAY);
    cvtColor(img2,gray2,COLOR_BGR2GRAY);
    Ptr<SIFT> sift = SIFT::create();
    std::vector<KeyPoint> kp1,kp2;
    Mat des1,des2;
    sift->detectAndCompute(gray1,Mat(),kp1,des1);
    qDebug()<<"time01= "<< time.elapsed()/1000.0<<"s";
    sift->detectAndCompute(gray2,Mat(),kp2,des2);
    qDebug()<<"time02= "<< time.elapsed()/1000.0<<"s";

    std::vector<KeyPoint> keypoints_obj,keypoints_scene;
    keypoints_obj = kp1;
    keypoints_scene = kp2;
    Mat descript_obj,descript_scene;
    descript_obj = des1;
    descript_scene = des2;
    FlannBasedMatcher fbmatcher;
    std::vector<DMatch> matches;
    fbmatcher.match(descript_obj, descript_scene, matches); //特征描述子匹配
    qDebug()<<"time1= "<< time.elapsed()/1000.0<<"s";
    //找出最优特征点
    double minDist = 1000;    //初始化最大最小距离
    double maxDist = 0;

    for (int i = 0; i < descript_obj.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist > maxDist)
        {
            maxDist = dist;
        }
        if (dist < minDist)
        {
            minDist = dist;
        }
    }
    qDebug()<<"maxDist:"<<maxDist;
    qDebug()<<"minDist:"<<minDist;

    std::vector<DMatch> goodMatches;
    for (int i = 0; i < descript_obj.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < 100) {
            goodMatches.push_back(matches[i]);
        }
    }

    if(goodMatches.size()<4){
        qDebug()<<"match fault";
        return;
    }

    std::vector<Point2f>point1, point2;
    for (int i = 0; i < goodMatches.size(); i++)
    {
        point1.push_back(kp1[goodMatches[i].queryIdx].pt);
        point2.push_back(kp2[goodMatches[i].trainIdx].pt);
    }

    Mat H = findHomography(point1, point2, RANSAC,5);
    std::vector<Point2f>cornerPoints1(4);
    std::vector<Point2f>cornerPoints2(4);
    cornerPoints1[0] = Point(0, 0);
    cornerPoints1[1] = Point(img1.cols, 0);
    cornerPoints1[2] = Point(img1.cols, img1.rows);
    cornerPoints1[3] = Point(0,img1.rows);
    perspectiveTransform(cornerPoints1, cornerPoints2, H);
    qDebug()<<"time2= "<< time.elapsed()/1000.0<<"s";

    Point resultPoint;
    resultPoint.x = cornerPoints2[0].x+45;
    resultPoint.y = cornerPoints2[0].y+45;
    qDebug()<<"resultPoint("<<resultPoint.x << ","<<resultPoint.y<<")";
    DD_mov(resultPoint.x,resultPoint.y);
    Sleep(20);
    DD_key(303,1);
    Sleep(20);
    DD_key(303,2);
    qDebug()<<"total time= "<< time.elapsed()/1000.0<<"s";

    //    //在原图上绘制出变换后的目标轮廓
    //    line(img2, cornerPoints2[0], cornerPoints2[1], Scalar(0,255,255), 4, 8, 0);
    //    line(img2, cornerPoints2[1], cornerPoints2[2], Scalar(0,255,255), 4, 8, 0);
    //    line(img2, cornerPoints2[2], cornerPoints2[3], Scalar(0,255,255), 4, 8, 0);
    //    line(img2, cornerPoints2[3], cornerPoints2[0], Scalar(0,255,255), 4, 8, 0);
    //    imshow("result", img2);
    //    waitKey(0);

}

void capture::GpDetcSurfFlann()
{
    QTime time;
    time.start();
    HWND hwnd = GetDesktopWindow();
    Mat screen = captureScreenMat(hwnd);
    qDebug()<<"screenshot time= "<< time.elapsed()/1000.0<<"s";

    if(screen.empty()){
        return  ;
    }
    cv::Mat src = screen.clone();
    cv::Mat temp = imread("F:\\CODE_PROJECT\\opencv002\\res\\lolkeg1.png");
    if(temp.empty()){
        qDebug() <<"sample is empty";
        return  ;
    }


    Mat img2 = src;
    Mat img1 = temp;
    Mat gray1,gray2,gaussian1,gaussian2;
    cvtColor(img1,gray1,COLOR_BGR2GRAY);
    cvtColor(img2,gray2,COLOR_BGR2GRAY);
    GaussianBlur(img1, gaussian1, Size(3, 3), 0);
    GaussianBlur(img2, gaussian2, Size(3, 3), 0);
    Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(1000);
    std::vector<KeyPoint> kp1,kp2;
    Mat des1,des2;
    surf->detectAndCompute(gaussian1,Mat(),kp1,des1);
    qDebug()<<"time01= "<< time.elapsed()/1000.0<<"s";
    surf->detectAndCompute(gaussian2,Mat(),kp2,des2);
    qDebug()<<"time02= "<< time.elapsed()/1000.0<<"s";

    std::vector<KeyPoint> keypoints_obj,keypoints_scene;
    keypoints_obj = kp1;
    keypoints_scene = kp2;
    Mat descript_obj,descript_scene;
    descript_obj = des1;
    descript_scene = des2;
    FlannBasedMatcher fbmatcher;
    std::vector<DMatch> matches;
    fbmatcher.match(descript_obj, descript_scene, matches); //特征描述子匹配
    qDebug()<<"time1= "<< time.elapsed()/1000.0<<"s";
    //找出最优特征点
    double minDist = 1000;    //初始化最大最小距离
    double maxDist = 0;

    for (int i = 0; i < descript_obj.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist > maxDist)
        {
            maxDist = dist;
        }
        if (dist < minDist)
        {
            minDist = dist;
        }
    }
    qDebug()<<"maxDist:"<<maxDist;
    qDebug()<<"minDist:"<<minDist;

    std::vector<DMatch> goodMatches;
    for (int i = 0; i < descript_obj.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < max((minDist*2),0.02)) {
            goodMatches.push_back(matches[i]);
        }
    }

    if(goodMatches.size()<4){
        qDebug()<<"match fault";
        return;
    }

    std::vector<Point2f>point1, point2;
    for (int i = 0; i < goodMatches.size(); i++)
    {
        point1.push_back(kp1[goodMatches[i].queryIdx].pt);
        point2.push_back(kp2[goodMatches[i].trainIdx].pt);
    }

    Mat H = findHomography(point1, point2, RANSAC,5);
    std::vector<Point2f>cornerPoints1(4);
    std::vector<Point2f>cornerPoints2(4);
    cornerPoints1[0] = Point(0, 0);
    cornerPoints1[1] = Point(img1.cols, 0);
    cornerPoints1[2] = Point(img1.cols, img1.rows);
    cornerPoints1[3] = Point(0,img1.rows);
    perspectiveTransform(cornerPoints1, cornerPoints2, H);
    qDebug()<<"time2= "<< time.elapsed()/1000.0<<"s";

    Point resultPoint;
    resultPoint.x = cornerPoints2[0].x+60;
    resultPoint.y = cornerPoints2[0].y+60;
    qDebug()<<"resultPoint("<<resultPoint.x << ","<<resultPoint.y<<")";
    DD_mov(resultPoint.x,resultPoint.y);
    Sleep(20);
    DD_key(303,1);
    Sleep(20);
    DD_key(303,2);
    qDebug()<<"total time= "<< time.elapsed()/1000.0<<"s";

    //        //在原图上绘制出变换后的目标轮廓
    //        line(img2, cornerPoints2[0]*2, cornerPoints2[1]*2, Scalar(0,255,255), 4, 8, 0);
    //        line(img2, cornerPoints2[1]*2, cornerPoints2[2]*2, Scalar(0,255,255), 4, 8, 0);
    //        line(img2, cornerPoints2[2]*2, cornerPoints2[3]*2, Scalar(0,255,255), 4, 8, 0);
    //        line(img2, cornerPoints2[3]*2, cornerPoints2[0]*2, Scalar(0,255,255), 4, 8, 0);
    //        imshow("result", img2);
    //        waitKey(0);

}

void capture::enemyMatch()
{
    QTime time;
    time.start();
    if(capMat.empty()){
        return  ;
    }
    cv::Mat src = capMat.clone();
    cv::Mat sample = imread("F:\\CODE_PROJECT\\opencv002\\res\\en2.png");
    cv::Mat mask = imread("F:\\CODE_PROJECT\\opencv002\\res\\en1.png");
    if(sample.empty()){
        qDebug() <<"sample is empty";
        return  ;
    }
    //预处理
    Mat src_down,sample_down,mask_down;
    pyrDown(src,src_down);
    pyrDown(sample,sample_down);
    pyrDown(mask,mask_down);
    //        pyrDown(src_down,src_down);
    //        pyrDown(sample_down,sample_down);
    //        pyrDown(mask_down,mask_down);
    //            imshow("src_down", src_down);
    //            waitKey(0);
    //            Mat src_gray,sample_gray,mask_gray;
    //            cvtColor(src_down,src_gray,COLOR_BGR2GRAY);
    //            cvtColor(sample_down,sample_gray,COLOR_BGR2GRAY);
    //        cvtColor(mask_down,mask_gray,COLOR_BGR2GRAY);
    //    Mat src_hsv;
    //    cvtColor(src_down,src_hsv,COLOR_BGR2HSV);
    //    Mat resulthsv;
    //    inRange(src_hsv,Scalar(0,242,51),Scalar(4,255,64),resulthsv);
    //        imshow("resulthsv", resulthsv);
    //        waitKey(0);
    //    Mat sample_hsv;
    //    cvtColor(sample,sample_hsv,COLOR_BGR2HSV);
    //    Mat sample_rthsv;
    //    inRange(sample_hsv,Scalar(0,242,51),Scalar(4,255,64),sample_rthsv);
    //        imshow("sample_rthsv", sample_rthsv);
    //        waitKey(0);

    //模板匹配
    Mat result;
    matchTemplate(src_down, sample_down, result, TM_CCORR_NORMED,mask_down);
    qDebug()<<"time2= "<< time.elapsed()/1000.0<<"s";

    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    qDebug()<<"maxVal"<<maxVal;
    qDebug()<<"maxLoc("<<maxLoc.x << ","<<maxLoc.y<<")";
    qDebug()<<"time3= "<< time.elapsed()/1000.0<<"s";

    //    if(maxVal<0.94){
    //        qDebug()<<"match failed";
    //        return;
    //    }
    //    else {
    //        qDebug()<<"total time= "<< time.elapsed()/1000.0<<"s";
    //        qDebug()<<"resultPoint("<<maxLoc.x*2 << ","<<maxLoc.y*2<<")";
    //        DD_mov(maxLoc.x*2+75,maxLoc.y*2+150);
    //        Sleep(20);
    //        DD_key(301,1);
    //        Sleep(20);
    //        DD_key(301,2);
    //        //        //结果绘制
    //        //        rectangle(src, Rect(maxLoc.x*2, maxLoc.y*2, sample.cols, sample.rows),Scalar(0, 255, 0), 2);
    //        //        imshow("template match result", src);
    //        //        waitKey(0);
    //    }

    //多目标匹配
    QList<QPoint> listResult;
    for (int i = 0; i < result.rows; i++)
    {
        for (int j = 0; j < result.cols; j++)
        {
            double val = result.at<float>(i, j);//灰度值

            //3*3邻域非极大值抑制
            if (val > 0.77)
            {
                if(i!=0 && j!=0 && i!=(result.rows-1) && j!=(result.cols-1)){
                    //当前像素的灰度值大于阈值且该像素是其3*3邻域最值时，判定其为目标
                    if (result.at<float>(i - 1, j - 1) < val &&
                            result.at<float>(i - 1, j) < val &&
                            result.at<float>(i - 1, j + 1) < val &&
                            result.at<float>(i, j - 1) < val &&
                            result.at<float>(i, j + 1) < val &&
                            result.at<float>(i + 1, j - 1) < val &&
                            result.at<float>(i + 1, j) < val &&
                            result.at<float>(i + 1, j + 1) < val)
                    {
                        qDebug()<<"result="<<val;
                        qDebug()<<"(x="<<j<<",y="<<i<<")";
                        listResult.append(QPoint(j*2, i*2));
                        //结果绘制
                        rectangle(src, Rect(j*2, i*2, sample.cols, sample.rows),Scalar(0, 255, 0), 2);

                        char text[10];
                        float score = result.at<float>(i, j);
                        sprintf_s(text, "%.2f",score);
                        putText(src, text, Point(j*2, i*2), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
                    }
                }
            }
        }
    }
    qDebug()<<"total time= "<< time.elapsed()/1000.0<<"s";
    qDebug()<<"total result= "<< listResult.size();
    imshow("multi template match result", src);
    waitKey(0);

    //    qDebug()<<"resultPoint("<<resultPoint.x << ","<<resultPoint.y<<")";
    //    DD_mov(resultPoint.x,resultPoint.y);
    //    Sleep(20);
    //    DD_key(303,1);
    //    Sleep(20);
    //    DD_key(303,2);

    //    // 框选结果
    //    cv::Mat draw = src.clone();
    //    Point matchF;
    //    matchF.x = maxLoc.x*2;
    //    matchF.y = maxLoc.y*2;
    //    rectangle(draw, matchF, Point(matchF.x + sample.cols, matchF.y + sample.rows), Scalar(0, 0, 255), 2, 8, 0);
    //    qDebug()<<"total_time= "<< time.elapsed()/1000.0<<"s";
    //    imshow("draw", draw);
    //    waitKey(0);
}



void capture::tsMatch()
{
    QTime time;
    time.start();
    if(capMat.empty()){
        return  ;
    }
    cv::Mat src = capMat.clone();
    cv::Mat sample = imread("F:\\CODE_PROJECT\\opencv002\\res\\lolkeg1.png");
    if(sample.empty()){
        qDebug() <<"sample is empty";
        return  ;
    }
    //预处理
    Mat src_down,sample_down;
    pyrDown(src,src_down);
    pyrDown(sample,sample_down);
    //模板匹配
    Mat result;
    matchTemplate(src_down, sample_down, result, TM_CCORR_NORMED);
    qDebug()<<"(sample_x="<<sample.cols<<",sample_y="<<sample.rows<<")";
    qDebug()<<"time0= "<< time.elapsed()/1000.0<<"s";
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    qDebug()<<"maxVal"<<maxVal;
    qDebug()<<"maxLoc("<<maxLoc.x << ","<<maxLoc.y<<")";
    qDebug()<<"time1= "<< time.elapsed()/1000.0<<"s";
    //多目标匹配
    QList<QPoint> listResult,finalResult;
    for (int i = 0; i < result.rows; i++)
    {
        for (int j = 0; j < result.cols; j++)
        {
            double val = result.at<float>(i, j);//灰度值

            //3*3邻域非极大值抑制
            if (val > 0.897)
            {
                if(i!=0 && j!=0 && i!=(result.rows-1) && j!=(result.cols-1)){
                    //当前像素的灰度值大于阈值且该像素是其3*3邻域最值时，判定其为目标
                    if (result.at<float>(i - 1, j - 1) < val &&
                            result.at<float>(i - 1, j) < val &&
                            result.at<float>(i - 1, j + 1) < val &&
                            result.at<float>(i, j - 1) < val &&
                            result.at<float>(i, j + 1) < val &&
                            result.at<float>(i + 1, j - 1) < val &&
                            result.at<float>(i + 1, j) < val &&
                            result.at<float>(i + 1, j + 1) < val)
                    {
                        qDebug()<<"result="<<val;
                        qDebug()<<"(x="<<j<<",y="<<i<<")";
                        listResult.append(QPoint(j*2, i*2));
                    }
                }
            }
        }
    }
    qDebug()<<"time2= "<< time.elapsed()/1000.0<<"s";
    //特征匹配
    qDebug()<<"temple result= "<< listResult.size();
    if(listResult.size()>0){
        for(int i=0;i<listResult.size();i++)
        {
            Mat imgRst = src(Rect(listResult.at(i).x(), listResult.at(i).y(), sample.cols, sample.rows));
            qDebug()<<listResult.at(i).x()<<listResult.at(i).y()<<sample.cols<<sample.rows;
            //            imshow(" match ", imgRst);
            //            waitKey(0);
            std::vector<KeyPoint> kp;
            Mat des;
            cvtColor(imgRst,imgRst,COLOR_RGB2GRAY);
            surf_detector->detectAndCompute(imgRst,Mat(),kp,des);
            qDebug()<<"Total keyponits: "<<kp.size();
            Mat descript_obj,descript_scene;
            descript_obj = des;
            descript_scene = keg_des;
            FlannBasedMatcher fbmatcher;
            std::vector<DMatch> matches;
            fbmatcher.match(descript_obj, descript_scene, matches); //特征描述子匹配
            qDebug()<<"time2= "<< time.elapsed()/1000.0<<"s";
            //找出最优特征点
            double minDist = 1000;    //初始化最大最小距离
            double maxDist = 0;

            for (int i = 0; i < descript_obj.rows; i++)
            {
                double dist = matches[i].distance;
                if (dist > maxDist)
                {
                    maxDist = dist;
                }
                if (dist < minDist)
                {
                    minDist = dist;
                }
            }
            qDebug()<<"maxDist:"<<maxDist;
            qDebug()<<"minDist:"<<minDist;

            //            std::vector<DMatch> goodMatches;
            //            for (int i = 0; i < descript_obj.rows; i++)
            //            {
            //                double dist = matches[i].distance;
            //                if (dist < 0.18) {
            //                    goodMatches.push_back(matches[i]);
            //                }
            //            }
            //            qDebug()<<"goodMatches::"<<goodMatches.size();
            //            if(goodMatches.size()<2){
            //                qDebug()<<"match fault"<<i;
            //            }
            //            else {
            //                //结果绘制
            //                rectangle(src, Rect(listResult.at(i).x(), listResult.at(i).y(), sample.cols, sample.rows),Scalar(0, 255, 0), 2);
            //                char text[10];
            //                float score = result.at<float>(listResult.at(i).y()/2, listResult.at(i).x()/2);
            //                sprintf_s(text, "%.2f",score);
            //                putText(src, text, Point(listResult.at(i).x(), listResult.at(i).y()), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
            //            }

            if(minDist>0.2){
                qDebug()<<"match fault"<<i;
            }
            else {
                finalResult.append(QPoint(listResult.at(i).x(), listResult.at(i).y()));
                //结果绘制
                rectangle(src, Rect(listResult.at(i).x(), listResult.at(i).y(), sample.cols, sample.rows),Scalar(0, 255, 0), 2);
                char text[10];
                float score = result.at<float>(listResult.at(i).y()/2, listResult.at(i).x()/2);
                sprintf_s(text, "%.2f",score);
                putText(src, text, Point(listResult.at(i).x(), listResult.at(i).y()), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
            }
        }
        if(finalResult.size()>1)
        {
            double result = atan2(finalResult.at(1).y() - finalResult.at(0).y(), finalResult.at(1).x() - finalResult.at(0).x())*180/M_PI;//-180~180度 逆时针度数
            result = -result;
            if(result<0)result = result + 360;
            qDebug()<<"result angle"<<result;
            calDistance(finalResult.at(0),finalResult.at(1));
        }
    }
    qDebug()<<"temple result= "<< listResult.size();
    qDebug()<<"final result= "<< finalResult.size();
    qDebug()<<"total time= "<< time.elapsed()/1000.0<<"s";
    imshow("multi template match result", src);
    waitKey(0);
}




void capture::sendMatSlot(const Mat &image)
{
    capMat = image.clone();
    input= image.clone();
    qDebug()<<"sendMatSlot triggered";
    qDebug()<<"(x="<<image.cols<<",y="<<image.rows<<")";
}

