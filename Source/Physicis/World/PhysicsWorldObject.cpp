#include "PhysicsWorldObject.h"
#include "PhysicsWorld.h"

PhysicsWorldObject::PhysicsWorldObject(QObject* parent)
	: Component()
{
}

PhysicsWorldObject::~PhysicsWorldObject()
{
	// remove it from the world
	m_world->removeEntity(this);
}
