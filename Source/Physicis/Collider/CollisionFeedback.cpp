#include "CollisionFeedback.h"


CollisionFeedback::CollisionFeedback()
	: m_colliding(false),
	  m_distanceSqaured(0.0f)
{}

CollisionFeedback::CollisionFeedback( const bool collidingState, const float distanceSqaured )
	: m_colliding(collidingState),
	  m_distanceSqaured(distanceSqaured)
{}
