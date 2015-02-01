#pragma once
#include "PhysicsWorldConfig.h"
#include "PhysicsWorldObject.h"

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

