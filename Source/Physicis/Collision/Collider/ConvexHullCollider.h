#pragma once
#include <Physicis/Geometry/ConvexShape.h>
#include "ICollider.h"

class Scene;
class ConvexHullCollider : public ICollider
{
public:
	ConvexHullCollider(const vec3& center, const QVector<vec3>& vertices, 
		const QVector<vec3>& faces, Scene* scene);

	ConvexShape getGeometryShape() const;

	virtual BroadPhaseCollisionFeedback onBroadPhase(ICollider* other); 

	/// Get the extreme vertex in the given direction
	virtual vec3 getLocalSupportPoint(const vec3& dir, float margin = 0) const;

protected:
	virtual void init(){}

private:
	ConvexShape m_convexShape;
};

typedef QSharedPointer<ConvexHullCollider> ConvexHullColliderPtr;