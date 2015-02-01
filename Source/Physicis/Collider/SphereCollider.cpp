#include "SphereCollider.h"


SphereCollider::SphereCollider( const vec3& center, const float radius )
	: AbstractCollider(center)
{
	m_sphereShape = SphereShape(center, radius);
}

SphereShape SphereCollider::getGeometryShape() const
{
	return m_sphereShape;
}

CollisionFeedback SphereCollider::intersect( AbstractCollider* other )
{
	SphereCollider* sp = dynamic_cast<SphereCollider*>(other);
	float radiusSum = m_sphereShape.getRadius() + sp->getGeometryShape().getRadius();
	float centerDis = (m_center - sp->getCenter()).length();

	return CollisionFeedback(centerDis > radiusSum, centerDis - radiusSum);
}
