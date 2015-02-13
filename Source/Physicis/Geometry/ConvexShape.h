#pragma once
#include "IShape.h"

class ConvexShape : public IShape
{
public:
	/// Creates a convex shape with the given vertices
	ConvexShape(const QVector<vec3>& vertices);
	~ConvexShape();

	void setVertices(const QVector<vec3> vertices) { m_vertices = vertices; }
	QVector<vec3> getVertices() const { return m_vertices; }

	void setScale(const vec3& scale);

private:
	QVector<vec3> m_vertices;
};

