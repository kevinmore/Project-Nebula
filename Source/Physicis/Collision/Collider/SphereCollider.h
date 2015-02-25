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
	float getRadius() const;

	void setScale(const float scale);

	virtual BroadPhaseCollisionFeedback onBroadPhase(ICollider* other);
	/// Get the extreme vertex in the given direction
	virtual vec3 getLocalSupportPoint(const vec3& dir, float margin = 0) const;

protected:
	virtual void init();

private:
	SphereShape m_sphereShape;
};

typedef QSharedPointer<SphereCollider> SphereColliderPtr;