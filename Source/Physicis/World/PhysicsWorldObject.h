#pragma once
#include <QObject>
#include <Physicis/Geometry/AbstractShape.h>

class PhysicsWorld;
class PhysicsWorldObject : public QObject
{
public:
	PhysicsWorldObject(QObject* parent = 0);
	~PhysicsWorldObject() {}

	inline PhysicsWorld* getWorld() const;
	inline void setWorld(PhysicsWorld* world);

protected:
	PhysicsWorld* m_world;
};

