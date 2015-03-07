#pragma once
#include <Physicis/Geometry/BoxShape.h>
#include "ICollider.h"

class Scene;
class BoxCollider : public ICollider
{
public:
	BoxCollider(const vec3& center, const vec3& halfExtents);
	BoxShape getGeometryShape() const;

	void setHalfExtents(const vec3& halfExtents);
	vec3 getHalfExtents() const;
	vec3 getAABBMinLocal() const;
	vec3 getAABBMaxLocal() const;
	
	void setScale(const vec3& scale);

	virtual BroadPhaseCollisionFeedback onBroadPhase(ICollider* other);

	/// Get the extreme vertex in the given direction
	virtual vec3 getLocalSupportPoint(const vec3& dir, float margin = 0) const;

protected:
	virtual void init();

private:
	BoxShape m_boxShape;
};

typedef QSharedPointer<BoxCollider> BoxColliderPtr;