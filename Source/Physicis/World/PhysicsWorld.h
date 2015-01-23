#pragma once
#include "PhysicsWorldConfig.h"

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
private:
	PhysicsWorldConfig m_config;
	float m_time;
	bool m_locked;
};

