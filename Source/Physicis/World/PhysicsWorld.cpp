#include "PhysicsWorld.h"
#include "PhysicsWorldObject.inl"

PhysicsWorld::PhysicsWorld(const PhysicsWorldConfig& config)
	: m_config(config),
	  m_time(0.0f),
	  m_locked(false)
{
}

PhysicsWorld::~PhysicsWorld()
{
}

PhysicsWorldConfig PhysicsWorld::getConfig() const
{
	return m_config;
}

void PhysicsWorld::update(const float deltaTime)
{
	// don't update the physics world when it is locked
	if (m_locked)
	{
		return;
	}

	// update the physics objects
	foreach(PhysicsWorldObject* obj, m_objectList)
	{
		if(obj)
			obj->update(deltaTime);
	}
}

bool PhysicsWorld::isLocked()
{
	return m_locked;
}

void PhysicsWorld::lock()
{
	m_locked = true;
}

void PhysicsWorld::unlock()
{
	m_locked = false;
}

void PhysicsWorld::addEntity( PhysicsWorldObject* entity )
{
	lock();
	m_objectList << entity;
	entity->setWorld(this);
	unlock();
}

void PhysicsWorld::removeEntity( PhysicsWorldObject* entity )
{
	lock();
	m_objectList.removeOne(entity);
	unlock();
}
