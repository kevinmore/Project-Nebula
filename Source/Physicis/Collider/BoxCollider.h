#pragma once
#include <Physicis/Geometry/BoxShape.h>
#include "AbstractCollider.h"

class Scene;
class BoxCollider : public AbstractCollider
{
public:
	BoxCollider(const vec3& center, const vec3& halfExtents, Scene* scene);
	BoxShape getGeometryShape() const;

	void setHalfExtents(const vec3& halfExtents);

	virtual CollisionFeedback intersect(AbstractCollider* other);

protected:
	virtual void init();

private:
	BoxShape m_boxShape;
};

typedef QSharedPointer<BoxCollider> BoxColliderPtr;