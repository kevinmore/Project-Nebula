#include "ConvexShape.h"
#include <QFile>
extern "C" {
#include <libqhull/qhull_a.h>
}

ConvexShape::ConvexShape( const QVector<vec3>& vertices )
	: IShape(IShape::CONVEXBODY, vec3(0, 0, 0))

{
	// prepare a vertex buffer from the given vertices for QHull
	QVector<double> inputBuffer;
	foreach(vec3 v, vertices) inputBuffer << v.x() << v.y() << v.z();

	// generate a convex hull using QHull
	// use stdout and stderr as FILE pointer for debug
	//FILE* p = fopen("convexhull.txt", "w");
	qh_new_qhull(3, vertices.size(), inputBuffer.data(), 0, "qhull Ft", 0, 0);
	//fclose(p);

	// read the output to generate a vertex buffer and indices buffer for rendering
// 	QFile file("convexhull.txt");
// 	if (file.open(QIODevice::ReadOnly | QIODevice::Text))
// 	{
// 		QTextStream in(&file);
// 		uint lineNumber = 0;
// 
// 		while (!in.atEnd()) {
// 			++lineNumber;
// 			QString line = in.readLine();
// 			if (lineNumber == 2)
// 			{
// 				qDebug() << line;
// 
// 			}
// 		}
// 	}

	// retrieve all the vertices of the convex hull
	vertexT* list = qh vertex_list;
	m_vertices.reserve(qh num_vertices);
	while(list && list->point)
	{
		m_vertices << vec3(list->point[0], list->point[1], list->point[2]);
		list = list->next;
	}

	// free memory
	qh_freeqhull(!qh_ALL);
}

ConvexShape::~ConvexShape()
{}

void ConvexShape::setScale( const vec3& scale )
{
	foreach(vec3 vertex, m_vertices)
	{
		vertex.setX(vertex.x() * scale.x());
		vertex.setY(vertex.y() * scale.y());
		vertex.setZ(vertex.z() * scale.z());
	}
}
