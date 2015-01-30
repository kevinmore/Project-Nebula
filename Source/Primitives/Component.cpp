#include "Component.h"
#include <Primitives/GameObject.h>

Component::Component(int renderLayer)
	: m_actor(0),
	  m_renderLayer(renderLayer)
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

int Component::renderLayer() const
{
	return m_renderLayer;
}
