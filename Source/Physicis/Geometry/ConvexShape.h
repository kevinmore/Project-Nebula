#pragma once
#include "IShape.h"

class ConvexShape : public IShape
{
public:
	/// Creates a convex shape with the given vertices
	ConvexShape(const QVector<vec3>& vertices);

	~ConvexShape();

	QVector<vec3> getVertices() const { return m_vertices; }
	QVector<vec3> getFaces() const { return m_faces; }
	QVector<vec3> getRenderingVertices() const { return m_verticesAndCentrums; }

private:
	QVector<vec3> m_verticesAndCentrums;
	QVector<vec3> m_vertices;
	QVector<vec3> m_faces;
};

