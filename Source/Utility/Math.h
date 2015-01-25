#pragma once
#include <Utility/EngineCommon.h>
#include <assimp/types.h>

namespace Math
{
	// check if a float is not a number
	static bool isNaN(float var)
	{
		volatile float d = var;
		return d != d;
	}

	namespace Matrix3
	{
		/**
         * Sets the value of the matrix from inertia tensor values.
         */
        static void setInertiaTensorCoeffs(mat3& matIn, float ix, float iy, float iz,
            float ixy=0, float ixz=0, float iyz=0)
        {
			matIn.m[0][0] = ix;
			matIn.m[1][1] = iy;
			matIn.m[2][2] = iz;

			matIn.m[0][1] = matIn.m[1][0] = -ixy;
			matIn.m[0][2] = matIn.m[2][0] = -ixz;
			matIn.m[0][3] = matIn.m[3][0] = -iyz;
        }

		/**
         * Sets the matrix to be a diagonal matrix with the given
         * values along the leading diagonal.
         */
        static void setDiagonal(mat3& matIn, float a, float b, float c)
        {
            setInertiaTensorCoeffs(matIn, a, b, c);
        }

		/**
         * Sets the value of the matrix as an inertia tensor of
         * a rectangular block aligned with the body's coordinate
         * system with the given axis half-sizes and mass.
         */
        static void setBlockInertiaTensor(mat3& matIn, const vec3& halfSizes, float mass)
        {
            vec3 squares(halfSizes.x() * halfSizes.x(), halfSizes.y() * halfSizes.y(), halfSizes.z() * halfSizes.z());
            setInertiaTensorCoeffs(matIn, 0.3f*mass*(squares.y() + squares.z()),
                0.3f*mass*(squares.x() + squares.z()),
                0.3f*mass*(squares.x() + squares.y()));
        }

		 /**
         * Sets the matrix to be the inverse of the given matrix.
         *
         * @param m The matrix to invert and use to set this.
         */
        static void setInverse(mat3 &m)
        {
            float t4 = m.m[0][0]*m.m[1][1];
            float t6 = m.m[0][0]*m.m[1][2];
            float t8 = m.m[0][1]*m.m[1][0];
            float t10 = m.m[0][2]*m.m[1][0];
            float t12 = m.m[0][1]*m.m[2][0];
            float t14 = m.m[0][2]*m.m[2][0];

            // Calculate the determinant
            float t16 = (t4*m.m[2][2] - t6*m.m[2][1] - t8*m.m[2][2]+
                        t10*m.m[2][1] + t12*m.m[1][2] - t14*m.m[1][1]);

            // Make sure the determinant is non-zero.
            if (t16 == (float)0.0f) return;
            float t17 = 1/t16;

            m.m[0][0] = (m.m[1][1]*m.m[2][2]-m.m[1][2]*m.m[2][1])*t17;
            m.m[0][1] = -(m.m[0][1]*m.m[2][2]-m.m[0][2]*m.m[2][1])*t17;
            m.m[0][2] = (m.m[0][1]*m.m[1][2]-m.m[0][2]*m.m[1][1])*t17;
            m.m[1][0] = -(m.m[1][0]*m.m[2][2]-m.m[1][2]*m.m[2][0])*t17;
            m.m[1][1] = (m.m[0][0]*m.m[2][2]-t14)*t17;
            m.m[1][2] = -(t6-t10)*t17;
            m.m[2][0] = (m.m[1][0]*m.m[2][1]-m.m[1][1]*m.m[2][0])*t17;
            m.m[2][1] = -(m.m[0][0]*m.m[2][1]-t12)*t17;
            m.m[2][2] = (t4-t8)*t17;
        }

		/**
         * Sets this matrix to be the rotation matrix corresponding to
         * the given quaternion.
         */
        static void setOrientation(mat3& m, const quart &q)
        {
			vec4 v = q.toVector4D();

			m.m[0][0] = 1 - (2*v.y()*v.y() + 2*v.z()*v.z());
			m.m[0][1] = 2*v.x()*v.y() + 2*v.z()*v.w();
			m.m[0][2] = 2*v.x()*v.z() - 2*v.y()*v.w();
			m.m[1][0] = 2*v.x()*v.y() - 2*v.z()*v.w();
			m.m[1][1] = 1 - (2*v.x()*v.x()  + 2*v.z()*v.z());
			m.m[1][2] = 2*v.y()*v.z() + 2*v.x()*v.w();
			m.m[2][0] = 2*v.x()*v.z() + 2*v.y()*v.w();
			m.m[2][1] = 2*v.y()*v.z() - 2*v.x()*v.w();
			m.m[2][2] = 1 - (2*v.x()*v.x()  + 2*v.y()*v.y());
        }
	}
	

	static void decomposeMat4(mat4& matIn, vec3& scalingOut, QQuaternion& rotationOut, vec3& positionOut)
	{
		// extract translation
		positionOut.setX(matIn(0, 3));
		positionOut.setY(matIn(1, 3));
		positionOut.setZ(matIn(2, 3));

		// extract the rows of the matrix
		vec3 vRows[3] = {
			vec3(matIn(0, 0), matIn(1, 0), matIn(2, 0)),
			vec3(matIn(0, 1), matIn(1, 1), matIn(2, 1)),
			vec3(matIn(0, 2), matIn(1, 2), matIn(2, 2))
		};

		// extract the scaling factors
		scalingOut.setX(vRows[0].length());
		scalingOut.setY(vRows[1].length());
		scalingOut.setZ(vRows[2].length());

		// and the sign of the scaling
		if (matIn.determinant() < 0) {
			scalingOut.setX(-scalingOut.x());
			scalingOut.setY(-scalingOut.y());
			scalingOut.setZ(-scalingOut.z());
		}

		// and remove all scaling from the matrix
		if(scalingOut.x())
		{
			vRows[0] /= scalingOut.x();
		}
		if(scalingOut.y())
		{
			vRows[1] /= scalingOut.y();
		}
		if(scalingOut.z())
		{
			vRows[2] /= scalingOut.z();
		}

		// build a 3x3 rotation matrix
		aiMatrix3x3 m(vRows[0].x(),vRows[1].x(),vRows[2].x(),
					  vRows[0].y(),vRows[1].y(),vRows[2].y(),
					  vRows[0].z(),vRows[1].z(),vRows[2].z());

		aiQuaternion r(m);
		rotationOut.setScalar(r.w);
		rotationOut.setX(r.x);
		rotationOut.setY(r.y);
		rotationOut.setZ(r.z);
	}

	namespace Random
	{
		static float random()
		{
			return rand() / (float)RAND_MAX;
		}

		static float random( float fMin, float fMax )
		{
			if ( fMin > fMax ) std::swap( fMin, fMax );
			return ( random() * ( fMax - fMin ) ) + fMin;
		}

		static vec3 random( const vec3& vMin, const vec3& vMax )
		{
			return vec3(random(vMin.x(), vMax.x()), random(vMin.y(), vMax.y()), random(vMin.z(), vMax.z()));
		}

		inline vec3 randUnitVec3()
		{
			float x = ( random() * 2.0f ) - 1.0f;
			float y = ( random() * 2.0f ) - 1.0f;
			float z = ( random() * 2.0f ) - 1.0f;

			return vec3(x,y,z).normalized();
		}
	}
	
	namespace Spline
	{
		// Interpolates between from and to by fraction. fraction is clamped between 0 and 1.
		// When fraction = 0 returns from. When fraction = 1 return to. When fraction = 0.5 returns the average of from and to.
		static float lerp(float from, float to, float fraction)
		{
			return from + ((to-from) * fraction);
		}

		static vec3 lerp(vec3& from, vec3& to, float fraction)
		{
			return from + ((to-from) * fraction);
		}

		static mat4 lerp(mat4& from, mat4& to, float fraction)
		{
			// decompose
			vec3	    scaling_from;
			QQuaternion rotation_from;
			vec3	    position_from;
			decomposeMat4(from, scaling_from, rotation_from, position_from);

			vec3	    scaling_to;
			QQuaternion rotation_to;
			vec3	    position_to;
			decomposeMat4(to, scaling_to, rotation_to, position_to);

			// lerp for 3 components
			vec3 scale(lerp(scaling_from, scaling_to, fraction));
			QQuaternion rotation(QQuaternion::slerp(rotation_from, rotation_to, fraction));
			vec3 position(lerp(position_from, position_to, fraction));

			// compose the result
			mat4 scaleM;
			scaleM.scale(scale);
			mat4 rotationM;
			rotationM.rotate(rotation);
			mat4 translationM;
			translationM.translate(position);
			
			return translationM * rotationM * scaleM;;
		}

		// simple function to generate a vector of 2d Bezier curve points
		static QVector<vec2> makeBezier2D(const QVector<vec2>& anchors, float accuracy = 10000.0f)
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
					temp.push_back(vec2(lerp(anchors[j-1].x(), anchors[j].x(), i),
					lerp(anchors[j-1].y(), anchors[j].y(), i)));

				while(temp.size()>1)
				{
					QVector<vec2> temp2;

					for(int j=1; j<temp.size(); ++j)
						temp2.push_back(vec2(lerp(temp[j-1].x(), temp[j].x(), i),
						lerp(temp[j-1].y(), temp[j].y(), i)));
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
					temp.push_back(vec3(lerp(anchors[j-1].x(), anchors[j].x(), i),
					lerp(anchors[j-1].y(), anchors[j].y(), i),
					lerp(anchors[j-1].z(), anchors[j].z(), i)));

				while(temp.size()>1)
				{
					QVector<vec3> temp2;

					for(int j=1; j<temp.size(); ++j)
						temp2.push_back(vec3(lerp(temp[j-1].x(), temp[j].x(), i),
						lerp(temp[j-1].y(), temp[j].y(), i),
						lerp(temp[j-1].z(), temp[j].z(), i)));
					temp = temp2;
				}
				curvePoints.push_back(temp[0]);
			}

			return curvePoints;
		}

		static vec3 catlerp(vec3& p0, vec3& p1, vec3& p2, vec3& p3, float t)
		{
			vec3 out;

			float t2 = t * t;
			float t3 = t2 * t;

			out.setX(0.5f * ((2.0f * p1.x()) +
				(-p0.x() + p2.x()) * t +
				(2.0f * p0.x() - 5.0f * p1.x() + 4 * p2.x() - p3.x()) * t2 +
				(-p0.x() + 3.0f * p1.x() - 3.0f * p2.x() + p3.x()) * t3));

			out.setY(0.5f * ((2.0f * p1.y()) +
				(-p0.y() + p2.y()) * t +
				(2.0f * p0.y() - 5.0f * p1.y() + 4 * p2.y() - p3.y()) * t2 +
				(-p0.y() + 3.0f * p1.y() - 3.0f * p2.y() + p3.y()) * t3));

			out.setZ(0.5f * ((2.0f * p1.z()) +
				(-p0.z() + p2.z()) * t +
				(2.0f * p0.z() - 5.0f * p1.z() + 4 * p2.z() - p3.z()) * t2 +
				(-p0.z() + 3.0f * p1.z() - 3.0f * p2.z() + p3.z()) * t3));

			return out;
		}

		static QVector<vec3> makeCatMullRomSpline(QVector<vec3>& anchors, int numPoints = 10000)
		{
			if (anchors.size() < 4)
				return anchors;

			QVector<vec3> splinePoints;

			for (int i = 0; i < anchors.size() - 3; ++i)
			{
				for (int j = 0; j < numPoints; ++j)
				{
					splinePoints.push_back(catlerp(anchors[i], anchors[i + 1], anchors[i + 2], anchors[i + 3], (1.0f / numPoints) * j));
				}
			}

			splinePoints.push_back(anchors[anchors.size() - 2]);

			return splinePoints;
		}
	} // end of Spline namespace

	
	
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

	namespace Quaternion
	{
		const QQuaternion ZERO = QQuaternion(1, 0, 0, 0);
	}
}