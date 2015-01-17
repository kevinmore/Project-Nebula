#include "Component.h"
#include <Scene/GameObject.h>

Component::Component(bool renderable, int renderOrder)
	: m_actor(0),
	  m_renderable(renderable),
	  m_renderOrder(renderOrder)
{}

Component::~Component()
{}

GameObject* Component::gameObject() const
{
	return m_actor;
}

void Component::linkGameObject( GameObject* go )
{
	m_actor = go;
}

bool Component::isRenderable() const
{
	return m_renderable;
}

int Component::renderOrder() const
{
	return m_renderOrder;
}
