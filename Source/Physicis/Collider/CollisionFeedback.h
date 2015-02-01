#pragma once
#include <math.h>

class CollisionFeedback
{
public:
	CollisionFeedback();
	CollisionFeedback(const bool collidingState, const float distanceSqaured);

	inline float getDistance() const { return sqrt(m_distanceSqaured); }
	inline bool isColliding() const  { return m_colliding;}
private:
	const bool  m_colliding;
	const float m_distanceSqaured;
};

