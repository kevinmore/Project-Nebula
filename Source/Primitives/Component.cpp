#include "Component.h"
#include <Scene/GameObject.h>


GameObject* Component::gameObject() const
{
	return m_actor;
}

void Component::linkGameObject( GameObject* go )
{
	m_actor = go;
}

Component::Component()
	:m_actor(0)
{

}
