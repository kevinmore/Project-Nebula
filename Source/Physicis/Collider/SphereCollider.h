#pragma once
#include <Physicis/Geometry/SphereShape.h>
#include "AbstractCollider.h"

class SphereCollider : public AbstractCollider
{
public:
	SphereCollider(const vec3& center, const float radius);
	SphereShape getGeometryShape() const;

	virtual CollisionFeedback intersect(AbstractCollider* other);

private:
	SphereShape m_sphereShape;
};

