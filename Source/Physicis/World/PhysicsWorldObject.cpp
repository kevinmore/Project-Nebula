#include "PhysicsWorldObject.h"
#include "PhysicsWorld.h"

PhysicsWorldObject::PhysicsWorldObject(QObject* parent)
	: Component(),
	  m_world(NULL)
{
}

PhysicsWorldObject::~PhysicsWorldObject()
{
}
