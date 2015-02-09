#pragma once
#include "EPATriangle.h"

class EPAPolytope
{
friend class EPATriangle;

public:
	EPAPolytope();
	~EPAPolytope();

	void Clear();
	QVector<EPATriangle*>& GetTriangles() { return m_Triangles; }
	const QVector<EPATriangle*>& GetTriangles() const { return m_Triangles; }
	QVector<vec3>& GetVertices() { return m_Vertices; }
	const QVector<vec3>& GetVertices() const { return m_Vertices; }

	EPATriangle* PopAClosestTriangleToOriginFromHeap();

	bool AddTetrahedron(const vec3& p0, const vec3& p1, const vec3& p2, const vec3& p3);
	bool AddHexahedron(const vec3& p0, const vec3& p1, const vec3& p2, const vec3& w0, const vec3& w1);
	bool ExpandPolytopeWithNewPoint(const vec3& w, EPATriangle* pTriangleUsedToObtainW);

	static bool IsOriginInTetrahedron(const vec3& p1, const vec3& p2, const vec3& p3, const vec3& p4);

private:
	QVector<vec3> m_Vertices; // array of vertices constructing this polytope. 	
	QVector<EPATriangle*> m_Triangles; // array of EPATriangle constructing this polytope. 

	QVector<int> m_SilhouetteVertices;
	QVector<EPATriangle*> m_SilhouetteTriangles;
	QVector<EPAEdge*> m_SilhouetteEdges;
	QVector<vec3> m_SupportPointsA; // support points from object A in local coordinate
	QVector<vec3> m_SupportPointsB; // support points from object B in local coordinate
	int m_Count;
};

