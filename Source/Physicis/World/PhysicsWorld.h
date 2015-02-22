#pragma once
#include "PhysicsWorldConfig.h"
#include "PhysicsWorldObject.h"
#include "Physicis/Collision/Collider/ICollider.h"

struct CollisionPair
{
	ICollider* pColliderA;
	ICollider* pColliderB;
	vec3 contactPoint;
	uint contactCount;

	CollisionPair(ICollider* a, ICollider* b)
	{
		pColliderA = a;
		pColliderB = b;
		contactPoint = vec3(1000, 1000, 1000);
		contactCount = 0;
	}
};

typedef QSharedPointer<CollisionPair> CollisionPairPtr;

class NarrowPhaseCollisionFeedback;
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
	void addCollider(ICollider* collider);

	void generateCollisionPairs();

	int entitiesCount();

	void reset();

	float computeContactImpulseMagnitude(const NarrowPhaseCollisionFeedback* pCollisionInfo);

	void elasticCollisionResponse(RigidBody* rb1, RigidBody* rb2);

private:
	PhysicsWorldConfig m_config;
	float m_time;
	bool m_locked;
	QList<PhysicsWorldObject*> m_entityList;
	QVector<ICollider*> m_colliderList;
	QList<CollisionPairPtr> m_collisionPairs;

	void boarderCheck(RigidBody* rb);
};

