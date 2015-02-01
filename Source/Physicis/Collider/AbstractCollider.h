#pragma once
#include "CollisionFeedback.h"
#include <Utility/Math.h>
#include <Primitives/Component.h>

class AbstractCollider : Component
{
public:
	AbstractCollider(const vec3& center) 
		: Component(0)
	{ m_center = center; }

	inline vec3 getCenter() const { return m_center; }

	virtual CollisionFeedback intersect(const AbstractCollider& other) = 0;
	virtual QString className() { return "Collider"; }

protected:
	vec3 m_center;
};