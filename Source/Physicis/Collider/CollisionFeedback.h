#pragma once
class CollisionFeedback
{
public:
	CollisionFeedback();
	CollisionFeedback(const bool collidingState, const float distance);

	inline float getDistance() const { return m_distance; }
	inline bool isColliding() const  { return m_colliding;}
private:
	const bool  m_colliding;
	const float m_distance;
};

