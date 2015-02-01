#include "PhysicsWorldObject.h"
#include "PhysicsWorld.h"

PhysicsWorldObject::PhysicsWorldObject()
	: Component()
{
}

PhysicsWorldObject::~PhysicsWorldObject()
{
	// remove it from the world
	m_world->removeEntity(this);
}
