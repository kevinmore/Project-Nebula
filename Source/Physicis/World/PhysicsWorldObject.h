#pragma once
#include <QObject>

class PhysicsWorldObject : QObject
{
public:
	PhysicsWorldObject(QObject* parent = 0);
	~PhysicsWorldObject();
};

