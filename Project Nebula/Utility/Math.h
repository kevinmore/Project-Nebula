#pragma once
#include <Utility/DataTypes.h>
#include <assimp/types.h>
namespace Math
{
	// simple interpolation function
	static float interpolate(float n1, float n2, float fraction)
	{
		return n1 + ((n2-n1) * fraction);
	}

	// simple function to generate a vector of 2d Bezier curve points
	static QVector<vec2> makeBezier2D(const QVector<vec2>& anchors, float accuracy=10000.0)
	{
		if(anchors.size()<=2)
			return anchors;

		QVector<vec2> curvePoints;
		curvePoints.push_back(anchors[0]);
		const float stride = 1.0f / accuracy;
		for(float i = 0.0f; i < 1.0f; i += stride)
		{
			QVector<vec2> temp;
			for(int j=1; j<anchors.size(); ++j)
				temp.push_back(vec2(interpolate(anchors[j-1].x(), anchors[j].x(), i),
									interpolate(anchors[j-1].y(), anchors[j].y(), i)));

			while(temp.size()>1)
			{
				QVector<vec2> temp2;

				for(int j=1; j<temp.size(); ++j)
					temp2.push_back(vec2(interpolate(temp[j-1].x(), temp[j].x(), i),
										 interpolate(temp[j-1].y(), temp[j].y(), i)));
				temp = temp2;
			}
			curvePoints.push_back(temp[0]);
		}

		return curvePoints;
	}

	// simple function to generate a vector of 3d Bezier curve points
	static QVector<vec3> makeBezier3D(const QVector<vec3>& anchors, float accuracy=10000.0)
	{
		if(anchors.size()<=2)
			return anchors;

		QVector<vec3> curvePoints;
		curvePoints.push_back(anchors[0]);
		const float stride = 1.0f / accuracy;
		for(float i = 0.0f; i < 1.0f; i += stride)
		{
			QVector<vec3> temp;
			for(int j=1; j<anchors.size(); ++j)
				temp.push_back(vec3(interpolate(anchors[j-1].x(), anchors[j].x(), i),
									interpolate(anchors[j-1].y(), anchors[j].y(), i),
									interpolate(anchors[j-1].z(), anchors[j].z(), i)));

			while(temp.size()>1)
			{
				QVector<vec3> temp2;

				for(int j=1; j<temp.size(); ++j)
					temp2.push_back(vec3(interpolate(temp[j-1].x(), temp[j].x(), i),
										 interpolate(temp[j-1].y(), temp[j].y(), i),
										 interpolate(temp[j-1].z(), temp[j].z(), i)));
				temp = temp2;
			}
			curvePoints.push_back(temp[0]);
		}

		return curvePoints;
	}

	// Computes the quaternion that is equivalent to a given
	// vector in order:  roll-pitch-yaw.
	static QQuaternion QuaternionFromEuler(vec3& v)
	{
		double c1 = cos(v.z() * 0.5);
		double c2 = cos(v.y() * 0.5);
		double c3 = cos(v.x() * 0.5);
		double s1 = sin(v.z() * 0.5);
		double s2 = sin(v.y() * 0.5);
		double s3 = sin(v.x() * 0.5);

		return QQuaternion(c1*c2*c3 + s1*s2*s3, c1*c2*s3 - s1*s2*c3, c1*s2*c3 + s1*c2*s3, s1*c2*c3 - c1*s2*s3);
	}

	
	// return Euler angles in roll-pitch-yaw order.
	static vec3 QuaternionToEuler(QQuaternion& q)
	{
		vec3 out;
		const static double PI_OVER_2 = M_PI * 0.5;
		const static double EPSILON = 1e-10;
		double sqw, sqx, sqy, sqz;
		vec4 qVec4 = q.toVector4D();
		// quick conversion to Euler angles to give tilt to user
		sqw = qVec4.w()*qVec4.w();
		sqx = qVec4.x()*qVec4.x();
		sqy = qVec4.y()*qVec4.y();
		sqz = qVec4.y()*qVec4.y();

		out.setY(asin(2.0 * (qVec4.w()*qVec4.y() - qVec4.x()*qVec4.z())));

		if (PI_OVER_2 - fabs(out.y()) > EPSILON) 
		{
			out.setZ ((float)atan2(2.0 * (qVec4.x()*qVec4.y() + qVec4.w()*qVec4.z()), sqx - sqy - sqz + sqw));
			out.setX ((float)atan2(2.0 * (qVec4.w()*qVec4.x() + qVec4.y()*qVec4.z()), sqw - sqx - sqy + sqz));
		} 
		else 
		{
			// compute heading from local 'down' vector
			out.setX(0.0f);
			out.setZ((float)atan2(2*qVec4.y()*qVec4.z() - 2*qVec4.x()*qVec4.w(),	2*qVec4.x()*qVec4.z() + 2*qVec4.y()*qVec4.w()));
			// If facing down, reverse yaw
			if (out.y() < 0)
				out.setZ(M_PI - out.z());
		}

		return out;
	}

	// utility function to convert aiMatrix4x4 to QMatrix4x4
	static QMatrix4x4 convToQMat4(const aiMatrix4x4 * m)
	{
		return QMatrix4x4(m->a1, m->a2, m->a3, m->a4,
						  m->b1, m->b2, m->b3, m->b4,
						  m->c1, m->c2, m->c3, m->c4,
						  m->d1, m->d2, m->d3, m->d4);
	}


	static QMatrix4x4 convToQMat4(aiMatrix3x3 * m) 
	{
		return QMatrix4x4(m->a1, m->a2, m->a3, 0,
						  m->b1, m->b2, m->b3, 0,
						  m->c1, m->c2, m->c3, 0,
						  0,     0,     0,     1);

	}

	static aiMatrix4x4 convToAiMat4(const QMatrix4x4 &m)
	{
		return aiMatrix4x4( m(0, 0), m(0, 1), m(0, 2), m(0, 3),
							m(1, 0), m(1, 1), m(1, 2), m(1, 3),
							m(2, 0), m(2, 1), m(2, 2), m(2, 3),
							m(3, 0), m(3, 1), m(3, 2), m(3, 3));
	}

	static void inverseQMat4(QMatrix4x4 &m)
	{
		float det = m.determinant();
		if(det == 0.0f) 
		{
			// Matrix not invertible. Setting all elements to nan is not really
			// correct in a mathematical sense but it is easy to debug for the
			// programmer.
			/*const float nan = std::numeric_limits<float>::quiet_NaN();
			*this = Matrix4f(
				nan,nan,nan,nan,
				nan,nan,nan,nan,
				nan,nan,nan,nan,
				nan,nan,nan,nan);*/
			return;
		}
		qDebug() << "Input:" << endl << m;
		float invdet = 1.0f / det;

		QMatrix4x4 res;
	
		res(0, 0) = invdet  * (m(1, 1) * (m(2, 2) * m(3, 3) - m(2, 3) * m(3, 2) + m(1, 2) * (m(2, 3) * m(3, 1) - m(2, 1) * m(3, 3)) + m(1, 3) * (m(2, 1) * m(3, 2) - m(2, 2) * m(3, 1))));
		res(0, 1) = -invdet * (m(0, 1) * (m(2, 2) * m(3, 3) - m(2, 3) * m(3, 2) + m(0, 2) * (m(2, 3) * m(3, 1) - m(2, 1) * m(3, 3)) + m(0, 3) * (m(2, 1) * m(3, 2) - m(2, 2) * m(3, 1))));
		res(0, 2) = invdet  * (m(0, 1) * (m(1, 2) * m(3, 3) - m(1, 3) * m(3, 2) + m(0, 2) * (m(1, 3) * m(3, 1) - m(1, 1) * m(3, 3)) + m(0, 3) * (m(1, 1) * m(3, 2) - m(1, 2) * m(3, 1))));
		res(0, 3) = -invdet * (m(0, 1) * (m(1, 2) * m(2, 3) - m(1, 3) * m(2, 2) + m(0, 2) * (m(1, 3) * m(2, 1) - m(1, 1) * m(2, 3)) + m(0, 3) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))));
		res(1, 0) = -invdet * (m(1, 0) * (m(2, 2) * m(3, 3) - m(2, 3) * m(3, 2) + m(1, 2) * (m(2, 3) * m(3, 0) - m(2, 0) * m(3, 3)) + m(1, 3) * (m(2, 0) * m(3, 2) - m(2, 2) * m(3, 0))));
		res(1, 1) = invdet  * (m(0, 0) * (m(2, 2) * m(3, 3) - m(2, 3) * m(3, 2) + m(0, 2) * (m(2, 3) * m(3, 0) - m(2, 0) * m(3, 3)) + m(0, 3) * (m(2, 0) * m(3, 2) - m(2, 2) * m(3, 0))));
		res(1, 2) = -invdet * (m(0, 0) * (m(1, 2) * m(3, 3) - m(1, 3) * m(3, 2) + m(0, 2) * (m(1, 3) * m(3, 0) - m(1, 0) * m(3, 3)) + m(0, 3) * (m(1, 0) * m(3, 2) - m(1, 2) * m(3, 0))));
		res(1, 3) = invdet  * (m(0, 0) * (m(1, 2) * m(2, 3) - m(1, 3) * m(2, 2) + m(0, 2) * (m(1, 3) * m(2, 0) - m(1, 0) * m(2, 3)) + m(0, 3) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))));
		res(2, 0) = invdet  * (m(1, 0) * (m(2, 1) * m(3, 3) - m(2, 3) * m(3, 1) + m(1, 1) * (m(2, 3) * m(3, 0) - m(2, 0) * m(3, 3)) + m(1, 3) * (m(2, 0) * m(3, 1) - m(2, 1) * m(3, 0))));
		res(2, 1) = -invdet * (m(0, 0) * (m(2, 1) * m(3, 3) - m(2, 3) * m(3, 1) + m(0, 1) * (m(2, 3) * m(3, 0) - m(2, 0) * m(3, 3)) + m(0, 3) * (m(2, 0) * m(3, 1) - m(2, 1) * m(3, 0))));
		res(2, 2) = invdet  * (m(0, 0) * (m(1, 1) * m(3, 3) - m(1, 3) * m(3, 1) + m(0, 1) * (m(1, 3) * m(3, 0) - m(1, 0) * m(3, 3)) + m(0, 3) * (m(1, 0) * m(3, 1) - m(1, 1) * m(3, 0))));
		res(2, 3) = -invdet * (m(0, 0) * (m(1, 1) * m(2, 3) - m(1, 3) * m(2, 1) + m(0, 1) * (m(1, 3) * m(2, 0) - m(1, 0) * m(2, 3)) + m(0, 3) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0))));
		res(3, 0) = -invdet * (m(1, 0) * (m(2, 1) * m(3, 2) - m(2, 2) * m(3, 1) + m(1, 1) * (m(2, 2) * m(3, 0) - m(2, 0) * m(3, 2)) + m(1, 2) * (m(2, 0) * m(3, 1) - m(2, 1) * m(3, 0))));
		res(3, 1) = invdet  * (m(0, 0) * (m(2, 1) * m(3, 2) - m(2, 2) * m(3, 1) + m(0, 1) * (m(2, 2) * m(3, 0) - m(2, 0) * m(3, 2)) + m(0, 2) * (m(2, 0) * m(3, 1) - m(2, 1) * m(3, 0))));
		res(3, 2) = -invdet * (m(0, 0) * (m(1, 1) * m(3, 2) - m(1, 2) * m(3, 1) + m(0, 1) * (m(1, 2) * m(3, 0) - m(1, 0) * m(3, 2)) + m(0, 2) * (m(1, 0) * m(3, 1) - m(1, 1) * m(3, 0))));
		res(3, 3) = invdet  * (m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1) + m(0, 1) * (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)))); 
	
	

		qDebug() << "Output:" << endl << res;
		m = QMatrix4x4(res);
	}

	namespace Vector2D
	{
		const QVector2D ZERO = QVector2D(0.0f, 0.0f);

		const QVector2D UNIT_X = QVector2D(1.0f, 0.0f);
		const QVector2D UNIT_Y = QVector2D(0.0f, 1.0f);

		const QVector2D NEGATIVE_UNIT_X = QVector2D(-1.0f,  0.0f);
		const QVector2D NEGATIVE_UNIT_Y = QVector2D( 0.0f, -1.0f);

		const QVector2D UNIT_SCALE = QVector2D(1.0f, 1.0f);
	}

	namespace Vector3D
	{
		const QVector3D ZERO = QVector3D(0.0f, 0.0f, 0.0f);

		const QVector3D UNIT_X = QVector3D(1.0f, 0.0f, 0.0f);
		const QVector3D UNIT_Y = QVector3D(0.0f, 1.0f, 0.0f);
		const QVector3D UNIT_Z = QVector3D(0.0f, 0.0f, 1.0f);

		const QVector3D NEGATIVE_UNIT_X = QVector3D(-1.0f,  0.0f,  0.0f);
		const QVector3D NEGATIVE_UNIT_Y = QVector3D( 0.0f, -1.0f,  0.0f);
		const QVector3D NEGATIVE_UNIT_Z = QVector3D( 0.0f,  0.0f, -1.0f);

		const QVector3D UNIT_SCALE = QVector3D(1.0f, 1.0f, 1.0f);
	}

	namespace Vector4D
	{
		const QVector4D ZERO = QVector4D(0.0f, 0.0f, 0.0f, 0.0f);
	}
}