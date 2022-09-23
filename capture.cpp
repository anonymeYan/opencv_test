#include "capture.h"


static int thresh = 130;
static int max_count = 255;
static Mat img, img_gray;
static const char* output_title = "Harris Corner Dectction Result";
static void Harris_Demo(int, void *);

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
    if(capMat.empty()){
        return  ;
    }
    Mat gray;
    cvtColor(capMat,gray,COLOR_BGR2GRAY);
    Ptr<SIFT> sift = SIFT::create();
    std::vector<KeyPoint> kp;
    Mat des;
    sift->detectAndCompute(gray,Mat(),kp,des);
    qDebug()<<"Total keyponits: "<<kp.size();

    Mat kpImage;
    drawKeypoints(gray,kp,kpImage,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
    imshow("SIFT KeyPoints",kpImage);
    waitKey(0);

}

void capture::surfDetection()
{
    //    if(capMat.empty()){
    //        return  ;
    //    }
    //    Mat gray;
    //    cvtColor(capMat,gray,COLOR_BGR2GRAY);

    //    Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create();
    //    std::vector<KeyPoint> kp;
    //    Mat des;
    //    detector->detectAndCompute(gray,Mat(),kp,des);
    //    qDebug()<<"Total keyponits: "<<kp.size();

    //    Mat kpImage;
    //    drawKeypoints(gray,kp,kpImage,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
    //    imshow("SURF KeyPoints",kpImage);
    //    waitKey(0);
}


void capture::orbDetection()
{
    if(capMat.empty()){
        return  ;
    }
    Ptr<ORB> orb = ORB::create();
    Mat gray,kp_img;
    cvtColor(capMat,gray,COLOR_BGR2GRAY);
    std::vector<KeyPoint> kp;
    Mat des;
    orb->detectAndCompute(gray,Mat(),kp,des);
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
    cv::Mat src = capMat.clone();
    cv::Mat sample = imread("F:\\CODE_PROJECT\\opencv001\\res\\lol002.png");
    if(sample.empty()){
        qDebug() <<"sample is empty";
        return  ;
    }
    int col= sample.cols;
    int row= sample.rows;
    qDebug() <<"sample.cols:"<<col<<" sample.rows:"<<row;
    // 匹配
    cv::Mat result;
    matchTemplate(src, sample, result, TM_CCOEFF);
    // 归一化
    normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

    //获取单个目标
    double minValue; double maxValue; Point minLocation; Point maxLocation;
    Point matchLocation;
    minMaxLoc(result, &minValue, &maxValue, &minLocation, &maxLocation, Mat());
    matchLocation = maxLocation;
    // 框选结果
    cv::Mat draw = src.clone();
    rectangle(draw, matchLocation, Point(matchLocation.x + sample.cols, matchLocation.y + sample.rows), Scalar(0, 0, 255), 2, 8, 0);

    //    //获取多个目标
    //    cv::Mat draw = src.clone();
    //    float threshT = 0.91f;
    //    for(int i=0;i<result.rows;i++){
    //        for (int j=0;j<result.cols;j++) {
    //            if (result.at<float>(i,j)>threshT){
    //                rectangle(draw, Point(j,i), Point(j + sample.cols, i + sample.rows), Scalar(0, 255, 255), 2, 8, 0);
    //                qDebug()<<"axis("<<i<<","<<j<<")";
    //            }
    //        }
    //    }

    imshow("draw", draw);
    waitKey(0);
}


void capture::TMultiMatch()
{
    if(capMat.empty()){
        return  ;
    }
    cv::Mat src = capMat.clone();
    cv::Mat temp = imread("F:\\CODE_PROJECT\\opencv001\\res\\lolkeg.png");
    if(temp.empty()){
        qDebug() <<"sample is empty";
        return  ;
    }
    Mat src_gray, src_gaussian;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    GaussianBlur(src_gray, src_gaussian, Size(3, 3), 0);

    Mat temp_gray, temp_gaussian;
    cvtColor(temp, temp_gray, COLOR_BGR2GRAY);
    GaussianBlur(temp_gray, temp_gaussian, Size(3, 3), 0);

    Mat result;
    matchTemplate(src_gaussian, temp_gaussian, result, TM_CCOEFF_NORMED);
    normalize(result, result, 0, 1, NORM_MINMAX);
    qDebug()<<"(src_gaussian_x="<<src_gaussian.cols<<",src_gaussian_y="<<src_gaussian.rows<<")";
    qDebug()<<"(result_x="<<result.cols<<",result_y="<<result.rows<<")";
    qDebug()<<"(temp_x="<<temp.cols<<",temp_y="<<temp.rows<<")";

    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    qDebug()<<"maxLoc("<<maxLoc.x << ","<<maxLoc.y<<")";
    //多目标匹配
    double quality = 0.85;  //匹配质量（0~1），越接近1，匹配程度越高
    if (quality <= 0.0)quality = 0.0;
    if (quality >= 1.0)quality = 1.0;
    double thresh = maxVal * quality;//像素阈值
    for (int i = 0; i < result.rows; i++)
    {
        for (int j = 0; j < result.cols; j++)
        {
            double val = result.at<float>(i, j);//灰度值

            //3*3邻域非极大值抑制
            if (val > thresh)
            {
                //当前像素的灰度值大于阈值且该像素是其3*3邻域最大值时，判定其为目标
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
                    //结果绘制
                    rectangle(src, Rect(j, i, temp.cols, temp.rows),Scalar(0, 255, 0), 2);

                    char text[10];
                    float score = result.at<float>(i, j);
                    sprintf_s(text, "%.2f",score);
                    putText(src, text, Point(j, i), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
                }
            }
        }
    }
    imshow("multi template match result", src);
    waitKey(0);
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
    BFMatcher bf = BFMatcher(NORM_L1);
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




void capture::sendMatSlot(const Mat &image)
{
    capMat = image.clone();
    qDebug()<<"sendMatSlot triggered";
    qDebug()<<"(x="<<image.cols<<",y="<<image.rows<<")";
}

