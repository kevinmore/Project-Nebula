#include "CollisionFeedback.h"


CollisionFeedback::CollisionFeedback()
	: m_colliding(false),
	  m_distance(0.0f)
{}

CollisionFeedback::CollisionFeedback( const bool collidingState, const float distance )
	: m_colliding(collidingState),
	  m_distance(distance)
{}
