#pragma once
#include <QObject>
#include <Physicis/Geometry/AbstractShape.h>
#include <Primitives/Component.h>

class PhysicsWorld;
class PhysicsWorldObject : public Component
{
public:
	PhysicsWorldObject(QObject* parent = 0);
	~PhysicsWorldObject() {}

	virtual QString className() { return "PhysicsWorldObject"; }
	virtual void render(const float currentTime) {/*do noting*/}

	inline PhysicsWorld* getWorld() const;
	inline void setWorld(PhysicsWorld* world);

	virtual void update(const float dt) = 0;

protected:
	PhysicsWorld* m_world;
};

