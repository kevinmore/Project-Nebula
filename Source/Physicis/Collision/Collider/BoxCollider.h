#pragma once
#include <Physicis/Geometry/BoxShape.h>
#include "ICollider.h"

class Scene;
class BoxCollider : public ICollider
{
public:
	BoxCollider(const vec3& center, const vec3& halfExtents, Scene* scene);
	BoxShape getGeometryShape() const;

	void setHalfExtents(const vec3& halfExtents);

	virtual BroadPhaseCollisionFeedback intersect(ICollider* other);

protected:
	virtual void init();

private:
	BoxShape m_boxShape;
};

typedef QSharedPointer<BoxCollider> BoxColliderPtr;