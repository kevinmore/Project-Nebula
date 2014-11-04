#pragma once
#include <Utility/DataTypes.h>
#include <assimp/types.h>
namespace Math
{
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