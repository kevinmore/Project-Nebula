#pragma once
/************************************************************************/
/* Component that can be attached to a Game Object                      */
/************************************************************************/
#include <QSharedPointer>
class GameObject;
class Component
{
public:
	Component(bool renderable = false, int renderOrder = -1);
	virtual ~Component() = 0;

	virtual void render(const float currentTime) = 0;

	GameObject* gameObject() const;
	void linkGameObject(GameObject* go);

	bool isRenderable() const; 
	int renderOrder() const;

protected:
	GameObject* m_actor;
	bool m_renderable; // decides if this component will be rendered
	int m_renderOrder; // a component with a less renderOrder(e.g. 0) get rendered first
};