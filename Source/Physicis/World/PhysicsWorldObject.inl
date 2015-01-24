#include "PhysicsWorldObject.h"
#include "PhysicsWorld.h"

inline PhysicsWorld* PhysicsWorldObject::getWorld() const
{
	return m_world;
}

inline void PhysicsWorldObject::setWorld( PhysicsWorld* world )
{
	m_world = world;
}