#include "CollisionFeedback.h"


CollisionFeedback::CollisionFeedback( const bool collidingState, const float distance )
	: m_colliding(collidingState),
	  m_distance(distance)
{

}
