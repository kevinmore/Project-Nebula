#pragma once
#include <Physicis/Geometry/ConvexShape.h>
#include "ICollider.h"

class Scene;
class ConvexHullCollider : public ICollider
{
public:
	ConvexHullCollider(const vec3& center, const ConvexShape& shape, Scene* scene);

	ConvexShape getGeometryShape() const;
	void setScale(const vec3& scale);

	virtual BroadPhaseCollisionFeedback onBroadPhase(ICollider* other); 

	/// Get the extreme vertex in the given direction
	virtual vec3 getLocalSupportPoint(const vec3& dir, float margin = 0) const;

protected:
	virtual void init();

private:
	ConvexShape m_convexShape;
	vec3 m_scale;
};

typedef QSharedPointer<ConvexHullCollider> ConvexHullColliderPtr;