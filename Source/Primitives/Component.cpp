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
	connect(go, SIGNAL(transformChanged(const Transform&)), this, SLOT(syncTransform(const Transform&)));
}


void Component::dislinkGameObject()
{
	disconnect(m_actor, 0, this, 0);
	m_actor = NULL;
}

int Component::renderLayer() const
{
	return m_renderLayer;
}

void Component::setRenderLayer( const int layerID )
{
	m_renderLayer = layerID;
}

const Transform& Component::getTransform() const
{
	return m_actor->getTransform();
}