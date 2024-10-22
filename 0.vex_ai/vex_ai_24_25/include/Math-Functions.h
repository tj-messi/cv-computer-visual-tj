#pragma once
#include <vector>

namespace tjulib{

  struct graphPoint
  {
    double x, y;
  };
  struct Point {
      double x, y, angle;
  };
  class Math
  {
    private:
      const float rad_In;
    public:
      // CONVERSIONS (DISTANCE)
      Math(float rad_In):rad_In(rad_In){};
      float degToInch(float deg);
      float inchToDeg(float inch);

      // CONVERSIONS (ANGLE)
      static float getRadians(float deg);
      static float getDeg(float rad);
      static double getWrap2pi(double currentAngle);
      static double getWrap360(double currentAngle);

      // HELPER FUNCTIONS
      static float getHeading(float angle);
      static float compressAngle(float startAngle, float angle);
      static float clip(float number, float min, float max);

      // GEOMETRY FUNCTIONS
      static float dist(graphPoint point1, graphPoint point2);
      static bool linePoint(graphPoint linePoint1, graphPoint linePoint2, graphPoint point);
      static bool pointCircle(graphPoint point, graphPoint circleCenter, float cr);
      static bool lineCircle(graphPoint linePoint1, graphPoint linePoint2, graphPoint circleCenter, float r);

      static constexpr float velocityToVoltage = 12000/200;

      // 计算两点之间的向量
      Point vectorBetweenPoints(const Point& p1, const Point& p2);
      // 计算向量的模
      double magnitude(const Point& v);
      // 计算两个向量的点积
      double dotProduct(const Point& v1, const Point& v2);
      // 计算两个向量之间的夹角（锐角）
      double angleBetweenVectors(const Point& v1, const Point& v2);
      // 将弧度转换为角度
      double radiansToDegrees(double radians);
      // 计算Point1/Point2 与Point2/Point3连线的夹角（锐角）
      double angleBetweenLines(const Point& point1, const Point& point2, const Point& point3);



      // PURE PURSUIT
      std::vector<graphPoint> lineCircleIntersection(graphPoint circleCenter, float radius,
                                                            graphPoint linePoint1, graphPoint linePoint2);
  };
};

