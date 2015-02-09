#include <algorithm>
#include "EPAPolytope.h"

EPAPolytope::EPAPolytope()
	: m_Count(0)
{}

EPAPolytope::~EPAPolytope()
{
	Clear();
}

void EPAPolytope::Clear()
{
	for (int i = 0; i < m_Triangles.size(); i++ )
		delete m_Triangles[i];

	m_Count = 0;
	m_Triangles.clear();

	m_Vertices.clear();
	m_SilhouetteVertices.clear();
	m_SilhouetteTriangles.clear();
	m_SilhouetteEdges.clear();
	m_SupportPointsA.clear();
	m_SupportPointsB.clear();
}

EPATriangle* EPAPolytope::PopAClosestTriangleToOriginFromHeap()
{
	EPATriangle* pReturnTriangle = NULL;

	if ( m_Triangles.size() == 0 )
		return pReturnTriangle;

	float minDistSqr = FLT_MAX;

	for ( int i = 0; i < (int)m_Triangles.size(); i++ )
	{
		if ( !m_Triangles[i]->IsObsolete() && m_Triangles[i]->IsClosestPointInternal() )
		{
			if ( m_Triangles[i]->GetDistSqr() < minDistSqr )
			{
				minDistSqr = m_Triangles[i]->GetDistSqr();
				pReturnTriangle = m_Triangles[i];
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


bool EPAPolytope::AddTetrahedron( const vec3& p0, const vec3& p1, const vec3& p2, const vec3& p3 )
{
	int index[4];
	m_Vertices.push_back(p0);
	index[0] = (int)m_Vertices.size() - 1;

	m_Vertices.push_back(p1);
	index[1] = (int)m_Vertices.size() - 1;

	m_Vertices.push_back(p2);
	index[2] = (int)m_Vertices.size() - 1;

	m_Vertices.push_back(p3);
	index[3] = (int)m_Vertices.size() - 1;

	EPATriangle* pTri[4];

	if ( vec3::dotProduct(vec3::crossProduct(p1-p0, p2-p0), p0) > 0 ) // p0, p1, p2 winding in counter-clockwise
	{
		// tet must contain the origin.
		if ( vec3::dotProduct(vec3::crossProduct(p1-p0, p2-p0), p3) >= 0 )
			return false;

		pTri[0] = new EPATriangle(index[0], index[1], index[2]); Q_ASSERT(CheckWinding(p0, p1, p2));
		pTri[1] = new EPATriangle(index[0], index[3], index[1]); Q_ASSERT(CheckWinding(p0, p3, p1));
		pTri[2] = new EPATriangle(index[0], index[2], index[3]); Q_ASSERT(CheckWinding(p0, p2, p3));
		pTri[3] = new EPATriangle(index[1], index[3], index[2]); Q_ASSERT(CheckWinding(p1, p3, p2));
	}
	else // p0, p2, p1 winding in counter-clockwise
	{
		// tet must contain the origin.
		if ( vec3::dotProduct(vec3::crossProduct(p2-p0, p1-p0), p3) >= 0 )
			return false;

		pTri[0] = new EPATriangle(index[0], index[2], index[1]); Q_ASSERT(CheckWinding(p0, p2, p1));
		pTri[1] = new EPATriangle(index[0], index[3], index[2]); Q_ASSERT(CheckWinding(p0, p3, p2));
		pTri[2] = new EPATriangle(index[0], index[1], index[3]); Q_ASSERT(CheckWinding(p0, p1, p3));
		pTri[3] = new EPATriangle(index[2], index[3], index[1]); Q_ASSERT(CheckWinding(p2, p3, p1));		
	}

	// construct adjacency
	pTri[0]->m_AdjacentTriangles[0] = pTri[1];
	pTri[0]->m_AdjacentTriangles[1] = pTri[3];
	pTri[0]->m_AdjacentTriangles[2] = pTri[2];

	pTri[1]->m_AdjacentTriangles[0] = pTri[2];
	pTri[1]->m_AdjacentTriangles[1] = pTri[3];
	pTri[1]->m_AdjacentTriangles[2] = pTri[0];

	pTri[2]->m_AdjacentTriangles[0] = pTri[0];
	pTri[2]->m_AdjacentTriangles[1] = pTri[3];
	pTri[2]->m_AdjacentTriangles[2] = pTri[1];

	pTri[3]->m_AdjacentTriangles[0] = pTri[1];
	pTri[3]->m_AdjacentTriangles[1] = pTri[2];
	pTri[3]->m_AdjacentTriangles[2] = pTri[0];

	pTri[0]->m_Edges[0]->m_pPairEdge = pTri[1]->m_Edges[2];
	pTri[0]->m_Edges[1]->m_pPairEdge = pTri[3]->m_Edges[2];
	pTri[0]->m_Edges[2]->m_pPairEdge = pTri[2]->m_Edges[0];

	pTri[1]->m_Edges[0]->m_pPairEdge = pTri[2]->m_Edges[2];
	pTri[1]->m_Edges[1]->m_pPairEdge = pTri[3]->m_Edges[0];
	pTri[1]->m_Edges[2]->m_pPairEdge = pTri[0]->m_Edges[0];

	pTri[2]->m_Edges[0]->m_pPairEdge = pTri[0]->m_Edges[2];
	pTri[2]->m_Edges[1]->m_pPairEdge = pTri[3]->m_Edges[1];
	pTri[2]->m_Edges[2]->m_pPairEdge = pTri[1]->m_Edges[0];

	pTri[3]->m_Edges[0]->m_pPairEdge = pTri[1]->m_Edges[1];
	pTri[3]->m_Edges[1]->m_pPairEdge = pTri[2]->m_Edges[1];
	pTri[3]->m_Edges[2]->m_pPairEdge = pTri[0]->m_Edges[1];

	EPATriangleComparison compare;

	for ( int i = 0; i < 4; i++ )
	{
		pTri[i]->ComputeClosestPointToOrigin(*this);

		pTri[i]->m_Index = m_Count++;

		m_Triangles.push_back(pTri[i]);
		std::push_heap(m_Triangles.begin(), m_Triangles.end(), compare);
	}

	return true;
}



// p0, p1 and p2 form a triangle. With this triangle, p and q create two tetrahedrons. 
// w0 is in the side of the direction which is calculated by (p1-p0).Cross(p2-p0)
// w1 is in the side of the direction which is calculated by -(p1-p0).Cross(p2-p0)
// By gluing these two tetrahedrons, hexahedron can be formed.
bool EPAPolytope::AddHexahedron( const vec3& p0, const vec3& p1, const vec3& p2, const vec3& w0, const vec3& w1 )
{
	if ( vec3::dotProduct(vec3::crossProduct(p1-p0, p2-p0), w0-p0) <= 0 )
		return false;

	if ( vec3::dotProduct(vec3::crossProduct(p1-p0, p2-p0), w1-p0) >= 0 )
		return false;

	int index[5];
	m_Vertices.push_back(p0);
	index[0] = (int)m_Vertices.size() - 1;

	m_Vertices.push_back(p1);
	index[1] = (int)m_Vertices.size() - 1;

	m_Vertices.push_back(p2);
	index[2] = (int)m_Vertices.size() - 1;

	m_Vertices.push_back(w0);
	index[3] = (int)m_Vertices.size() - 1;

	m_Vertices.push_back(w1);
	index[4] = (int)m_Vertices.size() - 1;

	EPATriangle* pTri[6];

	pTri[0] = new EPATriangle(index[0], index[1], index[3]); Q_ASSERT(CheckWinding(m_Vertices[index[0]], m_Vertices[index[1]], m_Vertices[index[3]]));
	pTri[1] = new EPATriangle(index[1], index[2], index[3]); Q_ASSERT(CheckWinding(m_Vertices[index[1]], m_Vertices[index[2]], m_Vertices[index[3]]));
	pTri[2] = new EPATriangle(index[2], index[0], index[3]); Q_ASSERT(CheckWinding(m_Vertices[index[2]], m_Vertices[index[0]], m_Vertices[index[3]]));
	pTri[3] = new EPATriangle(index[1], index[0], index[4]); Q_ASSERT(CheckWinding(m_Vertices[index[1]], m_Vertices[index[0]], m_Vertices[index[4]]));
	pTri[4] = new EPATriangle(index[2], index[1], index[4]); Q_ASSERT(CheckWinding(m_Vertices[index[2]], m_Vertices[index[1]], m_Vertices[index[4]]));
	pTri[5] = new EPATriangle(index[0], index[2], index[4]); Q_ASSERT(CheckWinding(m_Vertices[index[0]], m_Vertices[index[2]], m_Vertices[index[4]]));

	// construct adjacency
	pTri[0]->m_AdjacentTriangles[0] = pTri[3];
	pTri[0]->m_AdjacentTriangles[1] = pTri[1];
	pTri[0]->m_AdjacentTriangles[2] = pTri[2];

	pTri[1]->m_AdjacentTriangles[0] = pTri[4];
	pTri[1]->m_AdjacentTriangles[1] = pTri[2];
	pTri[1]->m_AdjacentTriangles[2] = pTri[0];

	pTri[2]->m_AdjacentTriangles[0] = pTri[5];
	pTri[2]->m_AdjacentTriangles[1] = pTri[0];
	pTri[2]->m_AdjacentTriangles[2] = pTri[1];

	pTri[3]->m_AdjacentTriangles[0] = pTri[0];
	pTri[3]->m_AdjacentTriangles[1] = pTri[5];
	pTri[3]->m_AdjacentTriangles[2] = pTri[4];

	pTri[4]->m_AdjacentTriangles[0] = pTri[1];
	pTri[4]->m_AdjacentTriangles[1] = pTri[3];
	pTri[4]->m_AdjacentTriangles[2] = pTri[5];

	pTri[5]->m_AdjacentTriangles[0] = pTri[2];
	pTri[5]->m_AdjacentTriangles[1] = pTri[4];
	pTri[5]->m_AdjacentTriangles[2] = pTri[3];

	pTri[0]->m_Edges[0]->m_pPairEdge = pTri[3]->m_Edges[0];
	pTri[0]->m_Edges[1]->m_pPairEdge = pTri[1]->m_Edges[2];
	pTri[0]->m_Edges[2]->m_pPairEdge = pTri[2]->m_Edges[1];

	pTri[1]->m_Edges[0]->m_pPairEdge = pTri[4]->m_Edges[0];
	pTri[1]->m_Edges[1]->m_pPairEdge = pTri[2]->m_Edges[2];
	pTri[1]->m_Edges[2]->m_pPairEdge = pTri[0]->m_Edges[1];

	pTri[2]->m_Edges[0]->m_pPairEdge = pTri[5]->m_Edges[0];
	pTri[2]->m_Edges[1]->m_pPairEdge = pTri[0]->m_Edges[2];
	pTri[2]->m_Edges[2]->m_pPairEdge = pTri[1]->m_Edges[1];

	pTri[3]->m_Edges[0]->m_pPairEdge = pTri[0]->m_Edges[0];
	pTri[3]->m_Edges[1]->m_pPairEdge = pTri[5]->m_Edges[2];
	pTri[3]->m_Edges[2]->m_pPairEdge = pTri[4]->m_Edges[1];

	pTri[4]->m_Edges[0]->m_pPairEdge = pTri[1]->m_Edges[0];
	pTri[4]->m_Edges[1]->m_pPairEdge = pTri[3]->m_Edges[2];
	pTri[4]->m_Edges[2]->m_pPairEdge = pTri[5]->m_Edges[1];

	pTri[5]->m_Edges[0]->m_pPairEdge = pTri[2]->m_Edges[0];
	pTri[5]->m_Edges[1]->m_pPairEdge = pTri[4]->m_Edges[2];
	pTri[5]->m_Edges[2]->m_pPairEdge = pTri[3]->m_Edges[1];

	EPATriangleComparison compare;

	for ( int i = 0; i < 6; i++ )
	{
		pTri[i]->ComputeClosestPointToOrigin(*this);

		pTri[i]->m_Index = m_Count++;

		for ( int j = 0; j < 3; j++ )
		{
			if ( !(pTri[i]->m_AdjacentTriangles[j] == pTri[i]->m_Edges[j]->m_pPairEdge->GetEPATriangle()) )
				return false;
		}

		m_Triangles.push_back(pTri[i]);
		std::push_heap(m_Triangles.begin(), m_Triangles.end(), compare);
	}

	return true;
}

bool EPAPolytope::ExpandPolytopeWithNewPoint( const vec3& w, EPATriangle* pTriangleUsedToObtainW )
{
	for (int i = 0; i < m_Triangles.size(); i++ )
	{
		m_Triangles[i]->m_bVisible = false;
	}

	m_SilhouetteVertices.clear();
	m_SilhouetteVertices.reserve(20);
	m_SilhouetteTriangles.clear();
	m_SilhouetteTriangles.reserve(20);
	m_SilhouetteEdges.clear();
	m_SilhouetteEdges.reserve(20);

	m_Vertices.push_back(w);
	int indexVertexW = (int)m_Vertices.size() - 1;

	Q_ASSERT(pTriangleUsedToObtainW->IsObsolete() == false);

	pTriangleUsedToObtainW->m_bVisible = true;
	pTriangleUsedToObtainW->SetObsolete(true);

	for ( int i = 0; i < (int)m_Triangles.size(); i++ )
	{
		if ( m_Triangles[i]->IsObsolete() )
			continue;

		int index = m_Triangles[i]->m_Index;
		bool b = m_Triangles[i]->IsVisibleFromPoint(w);
	}

	// 'Flood Fill Silhouette' algorithm to detect visible triangles and silhouette loop of edges from w.
	for ( int i = 0; i < 3; i++ )
		pTriangleUsedToObtainW->m_Edges[i]->m_pPairEdge->m_pEPATriangle->DoSilhouette(w, pTriangleUsedToObtainW->m_Edges[i], *this);

	Q_ASSERT(m_SilhouetteVertices.size() >= 3);
	Q_ASSERT(m_SilhouetteTriangles.size() >= 3);

	// Now, we create new triangles to patch the silhouette loop 
	int silhouetteSize = (int)m_SilhouetteVertices.size();

	for ( int i = 0; i < (int)m_Triangles.size(); i++ )
	{
		if ( m_Triangles[i]->IsObsolete() )
			continue;

		if ( m_Triangles[i]->m_bVisible )
			if ( m_Triangles[i]->IsVisibleFromPoint(w) != true )
				return false;
			else
				if ( m_Triangles[i]->IsVisibleFromPoint(w) != false )
					return false;
	}

	for ( int i = 0; i < (int)silhouetteSize; i++ )
	{
		if ( m_SilhouetteTriangles[i]->IsVisibleFromPoint(w) != false )
			return false;
	}

	QVector<EPATriangle*> newTriangles;
	newTriangles.reserve(silhouetteSize);

	EPATriangleComparison compare;

	for ( int i = 0; i < silhouetteSize; i++ )
	{
		int j = i+1 < silhouetteSize ? i+1 : 0;

		EPATriangle* pTri = new EPATriangle(indexVertexW, m_SilhouetteVertices[i], m_SilhouetteVertices[j]);
		newTriangles.push_back(pTri);
		pTri->ComputeClosestPointToOrigin(*this);
	}

	for ( int i = 0; i < silhouetteSize; i++ )
	{
		int j = (i+1 < silhouetteSize)? i+1 : 0;
		int k = (i-1 < 0)? silhouetteSize-1 : i-1;

		newTriangles[i]->m_AdjacentTriangles[2] = newTriangles[j];
		newTriangles[i]->m_Edges[2]->m_pPairEdge = newTriangles[j]->m_Edges[0];

		newTriangles[i]->m_AdjacentTriangles[0] = newTriangles[k];
		newTriangles[i]->m_Edges[0]->m_pPairEdge = newTriangles[k]->m_Edges[2];

		newTriangles[i]->m_AdjacentTriangles[1] = m_SilhouetteTriangles[i];
		newTriangles[i]->m_Edges[1]->m_pPairEdge = m_SilhouetteEdges[i];
		m_SilhouetteEdges[i]->m_pPairEdge = newTriangles[i]->m_Edges[1];
		m_SilhouetteTriangles[i]->m_AdjacentTriangles[m_SilhouetteEdges[i]->m_IndexLocal] = newTriangles[i];
	}

	for ( int i = 0; i < silhouetteSize; i++ )
	{	
		newTriangles[i]->m_Index = m_Count++;

		m_Triangles.push_back(newTriangles[i]);
		std::push_heap(m_Triangles.begin(), m_Triangles.end(), compare);
	}

	for ( int i = 0; i < GetTriangles().size(); i++ )
	{
		if ( !GetTriangles()[i]->IsObsolete() )
		{
			for ( int j = 0; j < 3; j++ )
			{
				EPAEdge* edge = GetTriangles()[i]->GetEdge(j);
				Q_ASSERT(edge->GetIndexVertex(0) == edge->m_pPairEdge->GetIndexVertex(1));
				Q_ASSERT(edge->GetIndexVertex(1) == edge->m_pPairEdge->GetIndexVertex(0));
			}
		}
	}

	return true;
}

bool EPAPolytope::IsOriginInTetrahedron( const vec3& p1, const vec3& p2, const vec3& p3, const vec3& p4 )
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
