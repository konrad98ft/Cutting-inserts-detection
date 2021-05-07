int main()
{
	 / / Create a blank map for drawing images
	Mat image = Mat::zeros(480, 640, CV_8UC3);
	 / / Enter the fit point
	vector<Point> points;
	points.push_back(Point(48, 58));
	points.push_back(Point(105, 98));
	points.push_back(Point(155, 160));
	points.push_back(Point(212, 220));
	points.push_back(Point(248, 260));
	points.push_back(Point(320, 300));
	points.push_back(Point(350, 360));
	points.push_back(Point(412, 400));
 
	 / / draw the fit point to the blank map
	for (int i = 0; i < points.size(); i++)
	{
		circle(image, points[i], 5, Scalar(0, 0, 255), 2, 8, 0);
	}
 
	Vec4f line_para;
	fitLine(points, line_para, cv::DIST_L2, 0, 1e-2, 1e-2);
	cout << "line_para = " << line_para << std::endl;
 
	 / / Get point oblique point and slope
	Point point0;
	 Point0.x = line_para[2];//point on the line
	point0.y = line_para[3];
	 Double k = line_para[1] / line_para[0]; //slope
 
	 //calculate the endpoint of the line (y = k(x - x0) + y0)
	Point point1, point2;
	point1.x = 0;
	point1.y = k * (0 - point0.x) + point0.y;
	point2.x = 640;
	point2.y = k * (640 - point0.x) + point0.y;
 
	line(image, point1, point2, cv::Scalar(0, 255, 0), 2, 8, 0);
	imshow("image", image);
	waitKey(0);
	return 0;
}
