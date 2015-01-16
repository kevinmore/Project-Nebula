#include "Component.h"
#include <Scene/GameObject.h>

Component::Component()
	:m_actor(0)
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