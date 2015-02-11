#include "ConvexHullCollider.h"

ConvexHullCollider::ConvexHullCollider( const vec3& center, const QVector<vec3>& vertices, const QVector<vec3>& faces, Scene* scene )
	: ICollider(center, scene),
	  m_convexShape(vertices, faces)
{
	m_colliderType = ICollider::COLLIDER_CONVEXHULL;
}

ConvexShape ConvexHullCollider::getGeometryShape() const
{
	return m_convexShape;
}

vec3 ConvexHullCollider::getLocalSupportPoint( const vec3& dir, float margin /*= 0*/ ) const
{
	float maxDot = -FLT_MAX;
	vec3 supportPoint;

	for ( int i = 0; i < m_convexShape.getVertices().size(); ++i )
	{
		const vec3 vertex = m_convexShape.getVertices()[i];
		float dot = vec3::dotProduct(vertex, dir);

		if ( dot > maxDot )
		{
			supportPoint = vertex;
			maxDot = dot;
		}
	}

	return supportPoint;
}

BroadPhaseCollisionFeedback ConvexHullCollider::onBroadPhase( ICollider* other )
{
	/*do nothing, this collider is for narrow phase collision detection*/
	return BroadPhaseCollisionFeedback();
}
