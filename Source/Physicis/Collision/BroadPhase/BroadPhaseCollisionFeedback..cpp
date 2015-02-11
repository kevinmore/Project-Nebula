#include "BroadPhaseCollisionFeedback.h"


BroadPhaseCollisionFeedback::BroadPhaseCollisionFeedback()
	: m_colliding(false),
	  m_distanceSqaured(0.0f)
{}

BroadPhaseCollisionFeedback::BroadPhaseCollisionFeedback( const bool collidingState, const float distanceSqaured )
	: m_colliding(collidingState),
	  m_distanceSqaured(distanceSqaured)
{}
