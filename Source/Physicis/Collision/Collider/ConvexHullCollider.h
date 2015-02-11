#pragma once
#include <Physicis/Geometry/ConvexShape.h>
#include "ICollider.h"

class Scene;
class ConvexHullCollider : public ICollider
{
public:
	ConvexHullCollider(void);
	~ConvexHullCollider(void);

	virtual BroadPhaseCollisionFeedback onBroadPhase(ICollider* other) 
	{/*do nothing, this collider is for narrow phase collision detection*/}
};

