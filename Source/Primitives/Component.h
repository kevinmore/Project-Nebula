#pragma once
/************************************************************************/
/* Component that can be attached to a Game Object                      */
/************************************************************************/
#include <QSharedPointer>
class GameObject;
class Component
{
public:
	Component();
	virtual ~Component() = 0;

	GameObject* gameObject() const;
	void linkGameObject(GameObject* go);

protected:
	GameObject* m_actor;
};