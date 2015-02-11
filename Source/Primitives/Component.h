#pragma once
/************************************************************************/
/* Component that can be attached to a Game Object                      */
/************************************************************************/
#include <QSharedPointer>
class Transform;
class GameObject;
class Component : public QObject
{
public:
	Component(int renderLayer = -1);
	virtual ~Component() = 0;

	virtual void render(const float currentTime) = 0;
	virtual QString className() = 0;

	GameObject* gameObject() const;
	void linkGameObject(GameObject* go);
	void dislinkGameObject();
	const Transform& getTransform() const;

	int renderLayer() const;

protected:
	GameObject* m_actor;
	int m_renderLayer; // a component with a less renderOrder(e.g. 0) get rendered first
					   // render layer < 0 means not renderable
};