#pragma once
/************************************************************************/
/* Component that can be attached to a Game Object                      */
/************************************************************************/
class GameObject;
class Component
{
public:
	Component();
	~Component() {}

	GameObject* gameObject() const;
	void linkGameObject(GameObject* go);

protected:
	GameObject* m_actor;
};