#include "EPATriangle.h"
#include "EPAPolytope.h"


EPATriangle::EPATriangle()
	: m_bObsolete(false),
	  m_bVisible(false),
	  m_index(0)
{
	for ( int i = 0; i < 3; i++ )
		m_edges[i] = NULL;
}

EPATriangle::EPATriangle( int indexVertex0, int indexVertex1, int indexVertex2 )
{
	m_indicesVertex[0] = indexVertex0;
	m_indicesVertex[1] = indexVertex1;
	m_indicesVertex[2] = indexVertex2;

	for ( unsigned int i = 0; i < 3; i++ )
		m_edges[i] = new EPAEdge(this, i, m_indicesVertex[i], m_indicesVertex[(i+1) % 3]);
}


EPATriangle::~EPATriangle()
{
	for ( int i = 0; i < 3; i++ )
		delete m_edges[i];
}

bool EPATriangle::isClosestPointInternal() const
{
	return ( m_lambda1 >= 0.0 && m_lambda2 >= 0.0 && (m_lambda1 + m_lambda2) <= m_det);
}

bool EPATriangle::isVisibleFromPoint(const vec3& point) const
{
	return vec3::dotProduct(point, m_closestPointToOrigin) >= m_distSqrd;
}

bool EPATriangle::computeClosestPointToOrigin(const EPAPolytope& EPAPolytope)
{
	const vec3& p0 = EPAPolytope.m_vertices[m_indicesVertex[0]];
	const vec3& p1 = EPAPolytope.m_vertices[m_indicesVertex[1]];
	const vec3& p2 = EPAPolytope.m_vertices[m_indicesVertex[2]];

	vec3 v1 = p1 - p0;
	vec3 v2 = p2 - p0;

	float v1Dotv1 = vec3::dotProduct(v1, v1);
	float v1Dotv2 = vec3::dotProduct(v1, v2);
	float v2Dotv2 = vec3::dotProduct(v2, v2);
	float p0Dotv1 = vec3::dotProduct(p0, v1);
	float p0Dotv2 = vec3::dotProduct(p0, v2);

	m_det = v1Dotv1 * v2Dotv2 - v1Dotv2 * v1Dotv2;
	m_lambda1 = p0Dotv2 * v1Dotv2 - p0Dotv1 * v2Dotv2;
	m_lambda2 = p0Dotv1 * v1Dotv2 - p0Dotv2 * v1Dotv1;

	if ( m_det > 0.0 ) 
	{
		m_closestPointToOrigin = p0 + 1.0f / m_det * (m_lambda1 * v1 + m_lambda2 * v2);
		m_distSqrd = m_closestPointToOrigin.lengthSquared();

		return true;
	}

	return false;
}

vec3 EPATriangle::getClosestPointToOriginInSupportPntSpace( const QVector<vec3>& supportPoints ) const
{
	const vec3* sp[3];

	for (int i = 0; i < 3; i++ )
		sp[i] = &supportPoints[m_indicesVertex[i]];

	return (*sp[0]) + (1.0f/m_det) * (m_lambda1 * ((*sp[1]) - (*sp[0])) + m_lambda2 * ((*sp[2]) - (*sp[0])));
}

// Please note that edge doesn't belong to this triangle. It is from the neighbor triangle.
// edge->m_pEPATriangle is a neighbor triangle which called this function. 
// edge->m_pPairEdge belongs to this triangle. 
bool EPATriangle::doSilhouette(const vec3& w, EPAEdge* edge, EPAPolytope& EPAPolytope)
{
	int index = m_index;

	Q_ASSERT(edge != NULL);
	Q_ASSERT(edge->m_pPairEdge != NULL);
	Q_ASSERT(edge->m_pEPATriangle != NULL);

	if ( m_bObsolete )
		return true;

	if ( !isVisibleFromPoint(w) ) // if this triangle is not visible from point w
	{
		int indexVertex0 = edge->m_indexVertex[0];
		EPAPolytope.m_silhouetteVertices.push_back(indexVertex0);
		EPAPolytope.m_silhouetteTriangles.push_back(this);
		EPAPolytope.m_silhouetteEdges.push_back(edge->m_pPairEdge);
		return true;
	}
	else // if visible
	{
		m_bVisible = true;

		m_bObsolete = true;
		EPAEdge* myEdge = edge->m_pPairEdge;

		Q_ASSERT(m_edges[myEdge->m_indexLocal] == myEdge);

		int indexNextEdgeCCW = (myEdge->m_indexLocal + 1) % 3;
		Q_ASSERT(0 <= indexNextEdgeCCW && indexNextEdgeCCW < 3);
		m_edges[indexNextEdgeCCW]->m_pPairEdge->m_pEPATriangle->doSilhouette(w, m_edges[indexNextEdgeCCW], EPAPolytope);

		indexNextEdgeCCW = (indexNextEdgeCCW + 1) % 3;
		Q_ASSERT(0 <= indexNextEdgeCCW && indexNextEdgeCCW < 3);
		m_edges[indexNextEdgeCCW]->m_pPairEdge->m_pEPATriangle->doSilhouette(w, m_edges[indexNextEdgeCCW], EPAPolytope);
	}

	return true;
}

bool EPATriangle::operator<(const EPATriangle& other) const
{
	return m_distSqrd > other.m_distSqrd;
}