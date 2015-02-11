#include "ConvexShape.h"

ConvexShape::ConvexShape( const QVector<vec3>& vertices, const QVector<vec3>& faces )
	: IShape(IShape::CONVEXBODY, vec3(0, 0, 0)),
	  m_vertices(vertices),
	  m_faces(faces),
	  m_innderRadius(0.0f),
	  m_outterRadius(0.0f)
{}

float ConvexShape::getInnderRadius()
{
	if(m_innderRadius > 0.0f) return m_innderRadius;

	//Find the innerRadius
	//For each face, calculate its distance from the particle's center and find the min
	float minDistance = 100.0f;
	for (int i = 0; i < m_faces.size(); ++i)
	{
		vec3 p = m_vertices[m_faces[i][0]];

		vec3 a(m_vertices[m_faces[i][1]][0] - p[0],
			m_vertices[m_faces[i][1]][1] - p[1],
			m_vertices[m_faces[i][1]][2] - p[2]);

		vec3 b(m_vertices[m_faces[i][2]][0] - p[0],
			m_vertices[m_faces[i][2]][1] - p[1],
			m_vertices[m_faces[i][2]][2] - p[2]);

		vec3 normal = vec3::crossProduct(a, b).normalized();

		float faceDistance = fabs(vec3::dotProduct(normal, p));

		minDistance = qMin(minDistance, faceDistance);
	}

	m_innderRadius = minDistance;
	return minDistance;
}

float ConvexShape::getOuttererRadius()
{
	if(m_outterRadius > 0.0f) return m_outterRadius;

	//Find the circumRadius
	//It's just the farthest vertex from the particle's center
	float maxDistance = 0.0f;
	for (int i = 0; i < m_vertices.size(); ++i)
	{
		maxDistance = qMax(maxDistance, m_vertices[i].lengthSquared());
	}

	m_outterRadius = qSqrt(maxDistance);

	return m_outterRadius;
}

