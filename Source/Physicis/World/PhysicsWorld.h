#pragma once
#include "PhysicsWorldConfig.h"
#include "PhysicsWorldObject.h"
#include <Physicis/Entity/BoxRigidBody.h>
#include <Physicis/Geometry/BoxShape.h>
#include <Physicis/Geometry/SphereShape.h>

class PhysicsWorld
{
public:
	PhysicsWorld(const PhysicsWorldConfig& config);
	~PhysicsWorld();

	PhysicsWorldConfig getConfig() const;

	void update(const float currentTime);
	
	bool isLocked();
	void lock();
	void unlock();

	void addEntity(PhysicsWorldObject* entity);
	void removeEntity(PhysicsWorldObject* entity);

private:
	PhysicsWorldConfig m_config;
	float m_time;
	bool m_locked;
	QList<PhysicsWorldObject*> m_objectList;
};

