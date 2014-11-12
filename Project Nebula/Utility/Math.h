#pragma once
#include <Utility/DataTypes.h>
#include <assimp/types.h>

namespace Math
{
	// check if a float is not a number
	static bool isNaN(float var)
	{
		volatile float d = var;
		return d != d;
	}

	namespace Spline
	{
		// simple interpolation function
		static float interpolate(float n1, float n2, float fraction)
		{
			return n1 + ((n2-n1) * fraction);
		}

		static vec2 interpolate(vec2& v1, vec2& v2, float fraction)
		{
			return vec2(interpolate(v1.x(), v2.x(), fraction),
				        interpolate(v1.y(), v2.y(), fraction));
		}

		static vec3 interpolate(vec3& v1, vec3& v2, float fraction)
		{
			return vec3(interpolate(v1.x(), v2.x(), fraction),
						interpolate(v1.y(), v2.y(), fraction),
						interpolate(v1.z(), v2.z(), fraction));
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
	}
	
	// EulerAngle structure
	struct EulerAngle
	{
		float m_fRoll, m_fPitch, m_fYaw;

		EulerAngle()
		{
			m_fRoll  = 0.0f;
			m_fPitch = 0.0f;
			m_fYaw   = 0.0f;
		}

		EulerAngle(float roll, float pith, float yaw)
		{
			m_fRoll  = roll;
			m_fPitch = pith;
			m_fYaw   = yaw;
		}
	};


	// Computes the quaternion that is equivalent to a given Euler Angle
	static QQuaternion QuaternionFromEuler(EulerAngle& ea)
	{

 		return QQuaternion::fromAxisAndAngle(vec3(0,0,1), ea.m_fRoll) *
			   QQuaternion::fromAxisAndAngle(vec3(0,1,0), ea.m_fYaw) *
 			   QQuaternion::fromAxisAndAngle(vec3(1,0,0), ea.m_fPitch);

	}
	
	// return Euler angles
	static EulerAngle QuaternionToEuler(QQuaternion& q)
	{
		EulerAngle out;

		vec4 v = q.toVector4D();
		float x = v.x();
		float y = v.y();
		float z = v.z();
		float w = v.w();

		out.m_fPitch  = qRadiansToDegrees(qAtan2(2 * (w * x + y * z) , 1 - 2 * (x * x + y * y)));
		out.m_fYaw = qRadiansToDegrees(qSin(2 * (w * y - x * z)));
		out.m_fRoll   = qRadiansToDegrees(qAtan2(2 * (w * z + x * y) , 1 - 2 * (y * y + z * z)));

		return out;
	}

	// utility function to convert aiMatrix4x4 to QMatrix4x4
	static QMatrix4x4 convToQMat4(const aiMatrix4x4& m)
	{
		return QMatrix4x4(m.a1, m.a2, m.a3, m.a4,
						  m.b1, m.b2, m.b3, m.b4,
						  m.c1, m.c2, m.c3, m.c4,
						  m.d1, m.d2, m.d3, m.d4);
	}


	static QMatrix4x4 convToQMat4(aiMatrix3x3& m) 
	{
		return QMatrix4x4(m.a1, m.a2, m.a3, 0,
						  m.b1, m.b2, m.b3, 0,
						  m.c1, m.c2, m.c3, 0,
						  0,     0,     0,     1);

	}

	static aiMatrix4x4 convToAiMat4(const QMatrix4x4 &m)
	{
		return aiMatrix4x4( m(0, 0), m(0, 1), m(0, 2), m(0, 3),
							m(1, 0), m(1, 1), m(1, 2), m(1, 3),
							m(2, 0), m(2, 1), m(2, 2), m(2, 3),
							m(3, 0), m(3, 1), m(3, 2), m(3, 3));
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