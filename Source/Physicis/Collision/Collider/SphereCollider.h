#pragma once
#include <Physicis/Geometry/SphereShape.h>
#include "ICollider.h"

class Scene;
class SphereCollider : public ICollider
{
public:
	SphereCollider(const vec3& center, const float radius, Scene* scene);
	SphereShape getGeometryShape() const;

	void setRadius(const float radius);

	virtual BroadPhaseCollisionFeedback onBroadPhase(ICollider* other);

protected:
	virtual void init();

private:
	SphereShape m_sphereShape;
};

typedef QSharedPointer<SphereCollider> SphereColliderPtr;