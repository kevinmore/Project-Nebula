#pragma once
#include "CollisionFeedback.h"
#include <Utility/Math.h>

class AbstractCollider
{
public:
	AbstractCollider(const vec3& center) { m_center = center; }

	inline vec3 getCenter() const { return m_center; }

	virtual CollisionFeedback intersect(const AbstractCollider& other) = 0;

protected:
	vec3 m_center;
};

