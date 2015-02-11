#pragma once
#include "PhysicsWorldConfig.h"
#include "PhysicsWorldObject.h"
#include "Physicis/Collision/Collider/ICollider.h"

class PhysicsWorld
{
public:
	PhysicsWorld(const PhysicsWorldConfig& config);
	~PhysicsWorld();

	PhysicsWorldConfig getConfig() const;

	void simulate(const float deltaTime);
	void handleCollisions();


	bool isLocked();
	void lock();
	void unlock();

	void addEntity(PhysicsWorldObject* entity);
	void removeEntity(PhysicsWorldObject* entity);

	int entitiesCount();

private:
	PhysicsWorldConfig m_config;
	float m_time;
	bool m_locked;
	QList<PhysicsWorldObject*> m_entityList;
	QVector<ICollider*> m_colliderList;
	void boarderCheck(RigidBody* rb);
};

