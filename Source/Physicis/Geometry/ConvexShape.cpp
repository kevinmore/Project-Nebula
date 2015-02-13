#include "ConvexShape.h"
extern "C" {
#include <libqhull/qhull_a.h>
}

ConvexShape::ConvexShape( const QVector<vec3>& vertices )
	: IShape(IShape::CONVEXBODY, vec3(0, 0, 0))

{
	// prepare a vertex buffer from the given vertices for QHull
	QVector<double> vertexBuffer;
	foreach(vec3 v, vertices) vertexBuffer << v.x() << v.y() << v.z();

	// generate a convex hull using QHull
	qh_new_qhull(3, vertices.size(), vertexBuffer.data(), 0, "qhull s", 0, 0);

	// retrieve all the vertices of the convex hull
	vertexT* list = qh vertex_list;
	m_vertices.reserve(qh num_vertices);
	while(list && list->point)
	{
		m_vertices << vec3(list->point[0], list->point[1], list->point[2]);
		list = list->next;
	}
}

ConvexShape::~ConvexShape()
{
	// free memory
	qh_freeqhull(!qh_ALL);
}

void ConvexShape::setScale( const vec3& scale )
{
	foreach(vec3 vertex, m_vertices)
	{
		vertex.setX(vertex.x() * scale.x());
		vertex.setY(vertex.y() * scale.y());
		vertex.setZ(vertex.z() * scale.z());
	}
}
