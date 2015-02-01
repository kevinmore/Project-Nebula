#pragma once
#include <Physicis/Geometry/SphereShape.h>
#include "AbstractCollider.h"

class Scene;
class SphereCollider : public AbstractCollider
{
public:
	SphereCollider(const vec3& center, const float radius, Scene* scene);
	SphereShape getGeometryShape() const;

	//virtual CollisionFeedback intersect(AbstractCollider* other);

protected:
	virtual void init();

private:
	SphereShape m_sphereShape;
};

typedef QSharedPointer<SphereCollider> SphereColliderPtr;