#pragma once
#include "IShape.h"

class ConvexShape : public IShape
{
public:
	/// Creates a convex shape with the given vertices and faces.
	ConvexShape(const QVector<vec3>& vertices, const QVector<vec3>& faces);

	void setVertices(const QVector<vec3> vertices) { m_vertices = vertices; }
	QVector<vec3> getVertices() const { return m_vertices; }

	void setFaces(const QVector<vec3> faces) { m_faces = faces; }
	QVector<vec3> getFaces() const { return m_faces; }


	float getInnderRadius();
	float getOuttererRadius();

private:
	QVector<vec3> m_vertices;
	QVector<vec3> m_faces;
	float m_innderRadius, m_outterRadius;
};

