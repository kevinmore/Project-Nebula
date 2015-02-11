#include <algorithm>
#include "Polytope.h"

Polytope::Polytope()
	: m_count(0)
{}

Polytope::~Polytope()
{
	clear();
}

void Polytope::clear()
{
	for (int i = 0; i < m_triangles.size(); i++ )
		SAFE_DELETE(m_triangles[i]);

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

	float minDistSqr = FLT_MAX;

	for ( int i = 0; i < (int)m_triangles.size(); i++ )
	{
		if ( !m_triangles[i]->isObsolete() && m_triangles[i]->isClosestPointInternal() )
		{
			if ( m_triangles[i]->getDistSqr() < minDistSqr )
			{
				minDistSqr = m_triangles[i]->getDistSqr();
				pReturnTriangle = m_triangles[i];
			}
		}
	}

	return pReturnTriangle;
}

static bool CheckWinding(const vec3& p0, const vec3& p1, const vec3& p2)
{
	vec3 temp = vec3::crossProduct(p1 - p0, p2 - p0);
	return vec3::dotProduct(temp, p0) > 0;
}


bool Polytope::addTetrahedron( const vec3& p0, const vec3& p1, const vec3& p2, const vec3& p3 )
{
	int index[4];
	m_vertices.push_back(p0);
	index[0] = (int)m_vertices.size() - 1;

	m_vertices.push_back(p1);
	index[1] = (int)m_vertices.size() - 1;

	m_vertices.push_back(p2);
	index[2] = (int)m_vertices.size() - 1;

	m_vertices.push_back(p3);
	index[3] = (int)m_vertices.size() - 1;

	Triangle* pTri[4];

	if ( vec3::dotProduct(vec3::crossProduct(p1-p0, p2-p0), p0) > 0 ) // p0, p1, p2 winding in counter-clockwise
	{
		// tet must contain the origin.
		if ( vec3::dotProduct(vec3::crossProduct(p1-p0, p2-p0), p3) >= 0 )
			return false;

		pTri[0] = new Triangle(index[0], index[1], index[2]); Q_ASSERT(CheckWinding(p0, p1, p2));
		pTri[1] = new Triangle(index[0], index[3], index[1]); Q_ASSERT(CheckWinding(p0, p3, p1));
		pTri[2] = new Triangle(index[0], index[2], index[3]); Q_ASSERT(CheckWinding(p0, p2, p3));
		pTri[3] = new Triangle(index[1], index[3], index[2]); Q_ASSERT(CheckWinding(p1, p3, p2));
	}
	else // p0, p2, p1 winding in counter-clockwise
	{
		// tet must contain the origin.
		if ( vec3::dotProduct(vec3::crossProduct(p2-p0, p1-p0), p3) >= 0 )
			return false;

		pTri[0] = new Triangle(index[0], index[2], index[1]); Q_ASSERT(CheckWinding(p0, p2, p1));
		pTri[1] = new Triangle(index[0], index[3], index[2]); Q_ASSERT(CheckWinding(p0, p3, p2));
		pTri[2] = new Triangle(index[0], index[1], index[3]); Q_ASSERT(CheckWinding(p0, p1, p3));
		pTri[3] = new Triangle(index[2], index[3], index[1]); Q_ASSERT(CheckWinding(p2, p3, p1));		
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

	EPATriangleComparison compare;

	for ( int i = 0; i < 4; i++ )
	{
		pTri[i]->computeClosestPointToOrigin(*this);

		pTri[i]->m_index = m_count++;

		m_triangles.push_back(pTri[i]);
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
	index[0] = (int)m_vertices.size() - 1;

	m_vertices.push_back(p1);
	index[1] = (int)m_vertices.size() - 1;

	m_vertices.push_back(p2);
	index[2] = (int)m_vertices.size() - 1;

	m_vertices.push_back(w0);
	index[3] = (int)m_vertices.size() - 1;

	m_vertices.push_back(w1);
	index[4] = (int)m_vertices.size() - 1;

	Triangle* pTri[6];

	pTri[0] = new Triangle(index[0], index[1], index[3]); Q_ASSERT(CheckWinding(m_vertices[index[0]], m_vertices[index[1]], m_vertices[index[3]]));
	pTri[1] = new Triangle(index[1], index[2], index[3]); Q_ASSERT(CheckWinding(m_vertices[index[1]], m_vertices[index[2]], m_vertices[index[3]]));
	pTri[2] = new Triangle(index[2], index[0], index[3]); Q_ASSERT(CheckWinding(m_vertices[index[2]], m_vertices[index[0]], m_vertices[index[3]]));
	pTri[3] = new Triangle(index[1], index[0], index[4]); Q_ASSERT(CheckWinding(m_vertices[index[1]], m_vertices[index[0]], m_vertices[index[4]]));
	pTri[4] = new Triangle(index[2], index[1], index[4]); Q_ASSERT(CheckWinding(m_vertices[index[2]], m_vertices[index[1]], m_vertices[index[4]]));
	pTri[5] = new Triangle(index[0], index[2], index[4]); Q_ASSERT(CheckWinding(m_vertices[index[0]], m_vertices[index[2]], m_vertices[index[4]]));

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

	EPATriangleComparison compare;

	for ( int i = 0; i < 6; i++ )
	{
		pTri[i]->computeClosestPointToOrigin(*this);

		pTri[i]->m_index = m_count++;

		for ( int j = 0; j < 3; j++ )
		{
			if ( !(pTri[i]->m_adjacentTriangles[j] == pTri[i]->m_edges[j]->m_pPairEdge->getTriangle()) )
				return false;
		}

		m_triangles.push_back(pTri[i]);
		std::push_heap(m_triangles.begin(), m_triangles.end(), compare);
	}

	return true;
}

bool Polytope::expandPolytopeWithNewPoint( const vec3& w, Triangle* pTriangleUsedToObtainW )
{
	for (int i = 0; i < m_triangles.size(); i++ )
	{
		m_triangles[i]->m_bVisible = false;
	}

	m_silhouetteVertices.clear();
	m_silhouetteVertices.reserve(20);
	m_silhouetteTriangles.clear();
	m_silhouetteTriangles.reserve(20);
	m_silhouetteEdges.clear();
	m_silhouetteEdges.reserve(20);

	m_vertices.push_back(w);
	int indexVertexW = (int)m_vertices.size() - 1;

	Q_ASSERT(pTriangleUsedToObtainW->isObsolete() == false);

	pTriangleUsedToObtainW->m_bVisible = true;
	pTriangleUsedToObtainW->setObsolete(true);

	for ( int i = 0; i < (int)m_triangles.size(); i++ )
	{
		if ( m_triangles[i]->isObsolete() )
			continue;

		int index = m_triangles[i]->m_index;
		bool b = m_triangles[i]->isVisibleFromPoint(w);
	}

	// 'Flood Fill Silhouette' algorithm to detect visible triangles and silhouette loop of edges from w.
	for ( int i = 0; i < 3; i++ )
		pTriangleUsedToObtainW->m_edges[i]->m_pPairEdge->m_pEPATriangle->doSilhouette(w, pTriangleUsedToObtainW->m_edges[i], *this);

	Q_ASSERT(m_silhouetteVertices.size() >= 3);
	Q_ASSERT(m_silhouetteTriangles.size() >= 3);

	// Now, we create new triangles to patch the silhouette loop 
	int silhouetteSize = (int)m_silhouetteVertices.size();

	for ( int i = 0; i < (int)m_triangles.size(); i++ )
	{
		if ( m_triangles[i]->isObsolete() )
			continue;

		if ( m_triangles[i]->m_bVisible )
			if ( m_triangles[i]->isVisibleFromPoint(w) != true )
				return false;
			else
				if ( m_triangles[i]->isVisibleFromPoint(w) != false )
					return false;
	}

	for ( int i = 0; i < (int)silhouetteSize; i++ )
	{
		if ( m_silhouetteTriangles[i]->isVisibleFromPoint(w) != false )
			return false;
	}

	QVector<Triangle*> newTriangles;
	newTriangles.reserve(silhouetteSize);

	EPATriangleComparison compare;

	for ( int i = 0; i < silhouetteSize; i++ )
	{
		int j = i+1 < silhouetteSize ? i+1 : 0;

		Triangle* pTri = new Triangle(indexVertexW, m_silhouetteVertices[i], m_silhouetteVertices[j]);
		newTriangles.push_back(pTri);
		pTri->computeClosestPointToOrigin(*this);
	}

	for ( int i = 0; i < silhouetteSize; i++ )
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

	for ( int i = 0; i < silhouetteSize; i++ )
	{	
		newTriangles[i]->m_index = m_count++;

		m_triangles.push_back(newTriangles[i]);
		std::push_heap(m_triangles.begin(), m_triangles.end(), compare);
	}

	for ( int i = 0; i < getTriangles().size(); i++ )
	{
		if ( !getTriangles()[i]->isObsolete() )
		{
			for ( int j = 0; j < 3; j++ )
			{
				Edge* edge = getTriangles()[i]->getEdge(j);
				Q_ASSERT(edge->getIndexVertex(0) == edge->m_pPairEdge->getIndexVertex(1));
				Q_ASSERT(edge->getIndexVertex(1) == edge->m_pPairEdge->getIndexVertex(0));
			}
		}
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
