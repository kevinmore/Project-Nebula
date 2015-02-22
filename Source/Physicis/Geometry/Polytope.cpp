#include <algorithm>
#include "Polytope.h"

static bool checkWinding(const vec3& p0, const vec3& p1, const vec3& p2)
{
	vec3 temp = vec3::crossProduct(p1 - p0, p2 - p0);
	return vec3::dotProduct(temp, p0) > 0;
}


Polytope::Polytope()
	: m_count(0)
{}

Polytope::~Polytope()
{
	clear();
}

void Polytope::clear()
{
	foreach(Triangle* pTri, m_triangles)
		SAFE_DELETE(pTri);

	m_count = 0;
	m_triangles.clear();

	m_vertices.clear();
	m_silhouetteVertices.clear();
	m_silhouetteTriangles.clear();
	m_silhouetteEdges.clear();
	m_supportPointsA.clear();
	m_supportPointsB.clear();
}

Triangle* Polytope::popAClosestTriangleToOriginFromHeap()
{
	Triangle* pReturnTriangle = NULL;

	if ( m_triangles.size() == 0 )
		return pReturnTriangle;

	float minDistSqrd = FLT_MAX;

	foreach(Triangle* pTri, m_triangles)
	{
		if ( !pTri->isObsolete() && pTri->isClosestPointInternal() && pTri->getDistSqrd() < minDistSqrd )
		{
			minDistSqrd = pTri->getDistSqrd();
			pReturnTriangle = pTri;
		}
	}

	return pReturnTriangle;
}

bool Polytope::addTetrahedron( const vec3& p0, const vec3& p1, const vec3& p2, const vec3& p3 )
{
	int index[4];
	m_vertices.push_back(p0);
	index[0] = m_vertices.size() - 1;

	m_vertices.push_back(p1);
	index[1] = m_vertices.size() - 1;

	m_vertices.push_back(p2);
	index[2] = m_vertices.size() - 1;

	m_vertices.push_back(p3);
	index[3] = m_vertices.size() - 1;

	Triangle* pTri[4];

	if ( vec3::dotProduct(vec3::crossProduct(p1-p0, p2-p0), p0) > 0 ) // p0, p1, p2 winding in counter-clockwise
	{
		// tet must contain the origin.
		if ( vec3::dotProduct(vec3::crossProduct(p1-p0, p2-p0), p3) >= 0 )
			return false;

		pTri[0] = new Triangle(index[0], index[1], index[2]); //assert(checkWinding(p0, p1, p2));
		pTri[1] = new Triangle(index[0], index[3], index[1]); //assert(checkWinding(p0, p3, p1));
		pTri[2] = new Triangle(index[0], index[2], index[3]); //assert(checkWinding(p0, p2, p3));
		pTri[3] = new Triangle(index[1], index[3], index[2]); //assert(checkWinding(p1, p3, p2));
	}
	else // p0, p2, p1 winding in counter-clockwise
	{
		// tet must contain the origin.
		if ( vec3::dotProduct(vec3::crossProduct(p2-p0, p1-p0), p3) >= 0 )
			return false;

		pTri[0] = new Triangle(index[0], index[2], index[1]); //assert(checkWinding(p0, p2, p1));
		pTri[1] = new Triangle(index[0], index[3], index[2]); //assert(checkWinding(p0, p3, p2));
		pTri[2] = new Triangle(index[0], index[1], index[3]); //assert(checkWinding(p0, p1, p3));
		pTri[3] = new Triangle(index[2], index[3], index[1]); //assert(checkWinding(p2, p3, p1));		
	}

	// construct adjacency
	pTri[0]->m_adjacentTriangles[0] = pTri[1];
	pTri[0]->m_adjacentTriangles[1] = pTri[3];
	pTri[0]->m_adjacentTriangles[2] = pTri[2];

	pTri[1]->m_adjacentTriangles[0] = pTri[2];
	pTri[1]->m_adjacentTriangles[1] = pTri[3];
	pTri[1]->m_adjacentTriangles[2] = pTri[0];

	pTri[2]->m_adjacentTriangles[0] = pTri[0];
	pTri[2]->m_adjacentTriangles[1] = pTri[3];
	pTri[2]->m_adjacentTriangles[2] = pTri[1];

	pTri[3]->m_adjacentTriangles[0] = pTri[1];
	pTri[3]->m_adjacentTriangles[1] = pTri[2];
	pTri[3]->m_adjacentTriangles[2] = pTri[0];

	pTri[0]->m_edges[0]->m_pPairEdge = pTri[1]->m_edges[2];
	pTri[0]->m_edges[1]->m_pPairEdge = pTri[3]->m_edges[2];
	pTri[0]->m_edges[2]->m_pPairEdge = pTri[2]->m_edges[0];

	pTri[1]->m_edges[0]->m_pPairEdge = pTri[2]->m_edges[2];
	pTri[1]->m_edges[1]->m_pPairEdge = pTri[3]->m_edges[0];
	pTri[1]->m_edges[2]->m_pPairEdge = pTri[0]->m_edges[0];

	pTri[2]->m_edges[0]->m_pPairEdge = pTri[0]->m_edges[2];
	pTri[2]->m_edges[1]->m_pPairEdge = pTri[3]->m_edges[1];
	pTri[2]->m_edges[2]->m_pPairEdge = pTri[1]->m_edges[0];

	pTri[3]->m_edges[0]->m_pPairEdge = pTri[1]->m_edges[1];
	pTri[3]->m_edges[1]->m_pPairEdge = pTri[2]->m_edges[1];
	pTri[3]->m_edges[2]->m_pPairEdge = pTri[0]->m_edges[1];

	TriangleComparison compare;

	for ( int i = 0; i < 4; ++i )
	{
		pTri[i]->computeClosestPointToOrigin(*this);

		m_triangles << pTri[i];
		std::push_heap(m_triangles.begin(), m_triangles.end(), compare);
	}

	return true;
}



// p0, p1 and p2 form a triangle. With this triangle, p and q create two tetrahedrons. 
// w0 is in the side of the direction which is calculated by (p1-p0).Cross(p2-p0)
// w1 is in the side of the direction which is calculated by -(p1-p0).Cross(p2-p0)
// By gluing these two tetrahedrons, hexahedron can be formed.
bool Polytope::addHexahedron( const vec3& p0, const vec3& p1, const vec3& p2, const vec3& w0, const vec3& w1 )
{
	if ( vec3::dotProduct(vec3::crossProduct(p1-p0, p2-p0), w0-p0) <= 0 )
		return false;

	if ( vec3::dotProduct(vec3::crossProduct(p1-p0, p2-p0), w1-p0) >= 0 )
		return false;

	int index[5];
	m_vertices.push_back(p0);
	index[0] = m_vertices.size() - 1;

	m_vertices.push_back(p1);
	index[1] = m_vertices.size() - 1;

	m_vertices.push_back(p2);
	index[2] = m_vertices.size() - 1;

	m_vertices.push_back(w0);
	index[3] = m_vertices.size() - 1;

	m_vertices.push_back(w1);
	index[4] = m_vertices.size() - 1;

	Triangle* pTri[6];

	pTri[0] = new Triangle(index[0], index[1], index[3]); assert(checkWinding(m_vertices[index[0]], m_vertices[index[1]], m_vertices[index[3]]));
	pTri[1] = new Triangle(index[1], index[2], index[3]); assert(checkWinding(m_vertices[index[1]], m_vertices[index[2]], m_vertices[index[3]]));
	pTri[2] = new Triangle(index[2], index[0], index[3]); assert(checkWinding(m_vertices[index[2]], m_vertices[index[0]], m_vertices[index[3]]));
	pTri[3] = new Triangle(index[1], index[0], index[4]); assert(checkWinding(m_vertices[index[1]], m_vertices[index[0]], m_vertices[index[4]]));
	pTri[4] = new Triangle(index[2], index[1], index[4]); assert(checkWinding(m_vertices[index[2]], m_vertices[index[1]], m_vertices[index[4]]));
	pTri[5] = new Triangle(index[0], index[2], index[4]); assert(checkWinding(m_vertices[index[0]], m_vertices[index[2]], m_vertices[index[4]]));

	// construct adjacency
	pTri[0]->m_adjacentTriangles[0] = pTri[3];
	pTri[0]->m_adjacentTriangles[1] = pTri[1];
	pTri[0]->m_adjacentTriangles[2] = pTri[2];

	pTri[1]->m_adjacentTriangles[0] = pTri[4];
	pTri[1]->m_adjacentTriangles[1] = pTri[2];
	pTri[1]->m_adjacentTriangles[2] = pTri[0];

	pTri[2]->m_adjacentTriangles[0] = pTri[5];
	pTri[2]->m_adjacentTriangles[1] = pTri[0];
	pTri[2]->m_adjacentTriangles[2] = pTri[1];

	pTri[3]->m_adjacentTriangles[0] = pTri[0];
	pTri[3]->m_adjacentTriangles[1] = pTri[5];
	pTri[3]->m_adjacentTriangles[2] = pTri[4];

	pTri[4]->m_adjacentTriangles[0] = pTri[1];
	pTri[4]->m_adjacentTriangles[1] = pTri[3];
	pTri[4]->m_adjacentTriangles[2] = pTri[5];

	pTri[5]->m_adjacentTriangles[0] = pTri[2];
	pTri[5]->m_adjacentTriangles[1] = pTri[4];
	pTri[5]->m_adjacentTriangles[2] = pTri[3];

	pTri[0]->m_edges[0]->m_pPairEdge = pTri[3]->m_edges[0];
	pTri[0]->m_edges[1]->m_pPairEdge = pTri[1]->m_edges[2];
	pTri[0]->m_edges[2]->m_pPairEdge = pTri[2]->m_edges[1];

	pTri[1]->m_edges[0]->m_pPairEdge = pTri[4]->m_edges[0];
	pTri[1]->m_edges[1]->m_pPairEdge = pTri[2]->m_edges[2];
	pTri[1]->m_edges[2]->m_pPairEdge = pTri[0]->m_edges[1];

	pTri[2]->m_edges[0]->m_pPairEdge = pTri[5]->m_edges[0];
	pTri[2]->m_edges[1]->m_pPairEdge = pTri[0]->m_edges[2];
	pTri[2]->m_edges[2]->m_pPairEdge = pTri[1]->m_edges[1];

	pTri[3]->m_edges[0]->m_pPairEdge = pTri[0]->m_edges[0];
	pTri[3]->m_edges[1]->m_pPairEdge = pTri[5]->m_edges[2];
	pTri[3]->m_edges[2]->m_pPairEdge = pTri[4]->m_edges[1];

	pTri[4]->m_edges[0]->m_pPairEdge = pTri[1]->m_edges[0];
	pTri[4]->m_edges[1]->m_pPairEdge = pTri[3]->m_edges[2];
	pTri[4]->m_edges[2]->m_pPairEdge = pTri[5]->m_edges[1];

	pTri[5]->m_edges[0]->m_pPairEdge = pTri[2]->m_edges[0];
	pTri[5]->m_edges[1]->m_pPairEdge = pTri[4]->m_edges[2];
	pTri[5]->m_edges[2]->m_pPairEdge = pTri[3]->m_edges[1];

	TriangleComparison compare;

	for ( int i = 0; i < 6; ++i )
	{
		pTri[i]->computeClosestPointToOrigin(*this);

		for ( int j = 0; j < 3; ++j )
		{
			if ( !(pTri[i]->m_adjacentTriangles[j] == pTri[i]->m_edges[j]->m_pPairEdge->getTriangle()) )
				return false;
		}

		m_triangles << pTri[i];
		std::push_heap(m_triangles.begin(), m_triangles.end(), compare);
	}

	return true;
}

bool Polytope::expandPolytopeWithNewPoint( const vec3& w, Triangle* pTriangleUsedToObtainW )
{
	m_silhouetteVertices.clear();
	m_silhouetteVertices.reserve(20);
	m_silhouetteTriangles.clear();
	m_silhouetteTriangles.reserve(20);
	m_silhouetteEdges.clear();
	m_silhouetteEdges.reserve(20);

	m_vertices.push_back(w);
	int indexVertexW = m_vertices.size() - 1;

	assert(pTriangleUsedToObtainW->isObsolete() == false);

	pTriangleUsedToObtainW->setObsolete(true);

	// 'Flood Fill Silhouette' algorithm to detect visible triangles and silhouette loop of edges from w.
	for ( int i = 0; i < 3; ++i )
	{
		Triangle* pTri = pTriangleUsedToObtainW->m_edges[i]->m_pPairEdge->m_pTriangle;
		//Triangle* pTri = pTriangleUsedToObtainW->m_adjacentTriangles[i];
		pTri->doSilhouette(w, pTriangleUsedToObtainW->m_edges[i], *this);
	}

	assert(m_silhouetteVertices.size() >= 3);
	assert(m_silhouetteTriangles.size() >= 3);

	// Now, we create new triangles to patch the silhouette loop 
	int silhouetteSize = m_silhouetteVertices.size();

	for ( int i = 0; i < silhouetteSize; ++i )
	{
		if ( m_silhouetteTriangles[i]->isVisibleFromPoint(w) != false )
			return false;
	}

	QVector<Triangle*> newTriangles;
	newTriangles.reserve(silhouetteSize);

	TriangleComparison compare;

	for ( int i = 0; i < silhouetteSize; ++i )
	{
		int j = i+1 < silhouetteSize ? i+1 : 0;

		Triangle* pTri = new Triangle(indexVertexW, m_silhouetteVertices[i], m_silhouetteVertices[j]);
		newTriangles.push_back(pTri);
		pTri->computeClosestPointToOrigin(*this);
	}

	for ( int i = 0; i < silhouetteSize; ++i )
	{
		int j = (i+1 < silhouetteSize)? i+1 : 0;
		int k = (i-1 < 0)? silhouetteSize-1 : i-1;

		newTriangles[i]->m_adjacentTriangles[2] = newTriangles[j];
		newTriangles[i]->m_edges[2]->m_pPairEdge = newTriangles[j]->m_edges[0];

		newTriangles[i]->m_adjacentTriangles[0] = newTriangles[k];
		newTriangles[i]->m_edges[0]->m_pPairEdge = newTriangles[k]->m_edges[2];

		newTriangles[i]->m_adjacentTriangles[1] = m_silhouetteTriangles[i];
		newTriangles[i]->m_edges[1]->m_pPairEdge = m_silhouetteEdges[i];
		m_silhouetteEdges[i]->m_pPairEdge = newTriangles[i]->m_edges[1];
		m_silhouetteTriangles[i]->m_adjacentTriangles[m_silhouetteEdges[i]->m_indexLocal] = newTriangles[i];
	}

	for ( int i = 0; i < silhouetteSize; ++i )
	{	
		m_triangles << newTriangles[i];
		std::push_heap(m_triangles.begin(), m_triangles.end(), compare);
	}

	return true;
}

bool Polytope::isOriginInTetrahedron( const vec3& p1, const vec3& p2, const vec3& p3, const vec3& p4 )
{
	float proj1, proj2;

	proj1 = vec3::dotProduct(vec3::crossProduct(p2-p1, p3-p1), p1);
	proj2 = vec3::dotProduct(vec3::crossProduct(p2-p1, p3-p1), p4);
	if (proj1 > 0.0 || proj2 > 0.0) 
		return false;

	proj1 = vec3::dotProduct(vec3::crossProduct(p4-p2, p3-p2), p1);
	proj2 = vec3::dotProduct(vec3::crossProduct(p4-p2, p3-p2), p2);
	if (proj1 > 0.0 || proj2 > 0.0)
		return false;

	proj1 = vec3::dotProduct(vec3::crossProduct(p4-p3, p1-p3), p2);
	proj2 = vec3::dotProduct(vec3::crossProduct(p4-p3, p1-p3), p3);
	if (proj1 > 0.0 || proj2 > 0.0)
		return false;

	proj1 = vec3::dotProduct(vec3::crossProduct(p2-p4, p1-p4), p3);
	proj2 = vec3::dotProduct(vec3::crossProduct(p2-p4, p1-p4), p4);
	if (proj1 > 0.0 || proj2 > 0.0)
		return false;

	return true;
}
