#include "EPATriangle.h"
#include "EPAPolytope.h"


EPATriangle::EPATriangle()
	: m_bObsolete(false),
	  m_bVisible(false),
	  m_Index(0)
{
	for ( int i = 0; i < 3; i++ )
		m_Edges[i] = NULL;
}

EPATriangle::EPATriangle( int indexVertex0, int indexVertex1, int indexVertex2 )
{
	m_IndicesVertex[0] = indexVertex0;
	m_IndicesVertex[1] = indexVertex1;
	m_IndicesVertex[2] = indexVertex2;

	for ( unsigned int i = 0; i < 3; i++ )
		m_Edges[i] = new EPAEdge(this, i, m_IndicesVertex[i], m_IndicesVertex[(i+1) % 3]);
}


EPATriangle::~EPATriangle()
{
	for ( int i = 0; i < 3; i++ )
		delete m_Edges[i];
}

bool EPATriangle::IsClosestPointInternal() const
{
	return ( m_Lambda1 >= 0.0 && m_Lambda2 >= 0.0 && (m_Lambda1 + m_Lambda2) <= m_Det);
}

bool EPATriangle::IsVisibleFromPoint(const vec3& point) const
{
	return vec3::dotProduct(point, m_ClosestPointToOrigin) >= m_DistSqr;
}

bool EPATriangle::ComputeClosestPointToOrigin(const EPAPolytope& EPAPolytope)
{
	const vec3& p0 = EPAPolytope.m_Vertices[m_IndicesVertex[0]];
	const vec3& p1 = EPAPolytope.m_Vertices[m_IndicesVertex[1]];
	const vec3& p2 = EPAPolytope.m_Vertices[m_IndicesVertex[2]];

	vec3 v1 = p1 - p0;
	vec3 v2 = p2 - p0;

	float v1Dotv1 = vec3::dotProduct(v1, v1);
	float v1Dotv2 = vec3::dotProduct(v1, v2);
	float v2Dotv2 = vec3::dotProduct(v2, v2);
	float p0Dotv1 = vec3::dotProduct(p0, v1);
	float p0Dotv2 = vec3::dotProduct(p0, v2);

	m_Det = v1Dotv1 * v2Dotv2 - v1Dotv2 * v1Dotv2;
	m_Lambda1 = p0Dotv2 * v1Dotv2 - p0Dotv1 * v2Dotv2;
	m_Lambda2 = p0Dotv1 * v1Dotv2 - p0Dotv2 * v1Dotv1;

	if ( m_Det > 0.0 ) 
	{
		m_ClosestPointToOrigin = p0 + 1.0f / m_Det * (m_Lambda1 * v1 + m_Lambda2 * v2);
		m_DistSqr = m_ClosestPointToOrigin.lengthSquared();

		return true;
	}

	return false;
}

vec3 EPATriangle::GetClosestPointToOriginInSupportPntSpace( const QVector<vec3>& supportPoints ) const
{
	const vec3* sp[3];

	for (int i = 0; i < 3; i++ )
		sp[i] = &supportPoints[m_IndicesVertex[i]];

	return (*sp[0]) + (1.0f/m_Det) * (m_Lambda1 * ((*sp[1]) - (*sp[0])) + m_Lambda2 * ((*sp[2]) - (*sp[0])));
}

// Please note that edge doesn't belong to this triangle. It is from the neighbor triangle.
// edge->m_pEPATriangle is a neighbor triangle which called this function. 
// edge->m_pPairEdge belongs to this triangle. 
bool EPATriangle::DoSilhouette(const vec3& w, EPAEdge* edge, EPAPolytope& EPAPolytope)
{
	int index = m_Index;

	Q_ASSERT(edge != NULL);
	Q_ASSERT(edge->m_pPairEdge != NULL);
	Q_ASSERT(edge->m_pEPATriangle != NULL);

	if ( m_bObsolete )
		return true;

	if ( !IsVisibleFromPoint(w) ) // if this triangle is not visible from point w
	{
		int indexVertex0 = edge->m_IndexVertex[0];
		EPAPolytope.m_SilhouetteVertices.push_back(indexVertex0);
		EPAPolytope.m_SilhouetteTriangles.push_back(this);
		EPAPolytope.m_SilhouetteEdges.push_back(edge->m_pPairEdge);
		return true;
	}
	else // if visible
	{
		m_bVisible = true;

		m_bObsolete = true;
		EPAEdge* myEdge = edge->m_pPairEdge;

		Q_ASSERT(m_Edges[myEdge->m_IndexLocal] == myEdge);

		int indexNextEdgeCCW = (myEdge->m_IndexLocal + 1) % 3;
		Q_ASSERT(0 <= indexNextEdgeCCW && indexNextEdgeCCW < 3);
		m_Edges[indexNextEdgeCCW]->m_pPairEdge->m_pEPATriangle->DoSilhouette(w, m_Edges[indexNextEdgeCCW], EPAPolytope);

		indexNextEdgeCCW = (indexNextEdgeCCW + 1) % 3;
		Q_ASSERT(0 <= indexNextEdgeCCW && indexNextEdgeCCW < 3);
		m_Edges[indexNextEdgeCCW]->m_pPairEdge->m_pEPATriangle->DoSilhouette(w, m_Edges[indexNextEdgeCCW], EPAPolytope);
	}

	return true;
}

bool EPATriangle::operator<(const EPATriangle& other) const
{
	return m_DistSqr > other.m_DistSqr;
}