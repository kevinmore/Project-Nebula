#pragma once
#include "EPATriangle.h"

class EPAPolytope
{
friend class EPATriangle;

public:
	EPAPolytope();
	~EPAPolytope();

	void clear();
	QVector<EPATriangle*>& getTriangles() { return m_triangles; }
	const QVector<EPATriangle*>& getTriangles() const { return m_triangles; }
	QVector<vec3>& getVertices() { return m_vertices; }
	const QVector<vec3>& getVertices() const { return m_vertices; }

	EPATriangle* popAClosestTriangleToOriginFromHeap();

	bool addTetrahedron(const vec3& p0, const vec3& p1, const vec3& p2, const vec3& p3);
	bool addHexahedron(const vec3& p0, const vec3& p1, const vec3& p2, const vec3& w0, const vec3& w1);
	bool expandPolytopeWithNewPoint(const vec3& w, EPATriangle* pTriangleUsedToObtainW);

	static bool isOriginInTetrahedron(const vec3& p1, const vec3& p2, const vec3& p3, const vec3& p4);

private:
	QVector<vec3> m_vertices; // array of vertices constructing this polytope. 	
	QVector<EPATriangle*> m_triangles; // array of EPATriangle constructing this polytope. 

	QVector<int> m_silhouetteVertices;
	QVector<EPATriangle*> m_silhouetteTriangles;
	QVector<EPAEdge*> m_silhouetteEdges;
	QVector<vec3> m_supportPointsA; // support points from object A in local coordinate
	QVector<vec3> m_supportPointsB; // support points from object B in local coordinate
	int m_count;
};

