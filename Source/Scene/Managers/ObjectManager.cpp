#include "ObjectManager.h"
#include <Scene/Scene.h>
#include <Animation/FK/FKController.h>
#include <Animation/IK/FABRIKSolver.h>
#include <Physicis/Entity/BoxRigidBody.h>
#include <Physicis/Entity/SphereRigidBody.h>

ObjectManager::ObjectManager(Scene* scene, QObject* parent)
	: QObject(parent),
	  m_scene(scene),
	  m_loadingFlag("Quality")
{}


ObjectManager::~ObjectManager() {}

void ObjectManager::registerGameObject( const QString& name, GameObjectPtr go )
{
	// add the game object into the map
	m_gameObjectMap[name] = go;

	// extract the renderable components
	foreach(ComponentPtr comp, go->getComponents())
	{
		addComponentToRenderQueue(comp);
	}

	// listen for further updates (attach and detach component)
	connect(go.data(), SIGNAL(componentAttached(ComponentPtr)), this, SLOT(addComponentToRenderQueue(ComponentPtr)));
	connect(go.data(), SIGNAL(componentDetached(ComponentPtr)), this, SLOT(removeComponentFromRenderQueue(ComponentPtr)));
}

GameObjectPtr ObjectManager::getGameObject( const QString& name )
{
	if(m_gameObjectMap.find(name) != m_gameObjectMap.end()) return m_gameObjectMap[name];
	else return GameObjectPtr();
}

void ObjectManager::renderAll(const float currentTime)
{
	// sync physics engine feedback
	foreach(GameObjectPtr go, m_gameObjectMap.values())
	{
		RigidBodyPtr rb = go->getComponent("RigidBody").dynamicCast<RigidBody>();
		if (rb)
		{
// 			QObjectList children = go->children();
// 			foreach(QObject* obj, children)
// 			{
// 				GameObject* child = dynamic_cast<GameObject*>(obj);
// 				ParticleSystemPtr ps = child->getComponent("ParticleSystem").dynamicCast<ParticleSystem>();
// 				rb->applyPointImpulse(ps->getLinearImpuse() * 0.8f, child->position() / 100.0f);
// 			}
 			//rb->applyAngularImpulse(vec3(0,0.1,0.1));
			//rb->applyPointImpulse(vec3(0.1, 0, 0), vec3(0.2, 0, 0.2));
			go->setPosition(rb->getPosition());
			go->setRotation(rb->getEularAngles());

// 			mat4 temp = rb->getTransformMatrix();
// 			temp.scale(go->scale());
// 			go->setTransformMatrix(temp);
		}
	}

	//int totalParticles = 0;
	foreach(ComponentPtr comp, m_renderQueue)
	{
		comp->render(currentTime);
		//totalParticles += comp.dynamicCast<ParticleSystem>()->getAliveParticles();
	}

	// print out the particles count
	//if(totalParticles) qDebug() << "Alive Particles:" << totalParticles;
}

void ObjectManager::clear()
{
	// clean up
	// since we are using QSharedPointer as the data stored in the containers here
	// there's no need to destroy each object individually,
	// they get destroyed automatically
	m_modelLoaders.clear();
	m_gameObjectMap.clear();
	m_renderQueue.clear();
}

GameObjectPtr ObjectManager::createGameObject( const QString& customName, GameObject* parent /*= 0*/ )
{

	// check if this object has the same name with another
	QString name = customName;
	int duplication = 0;
	foreach(QString key, m_gameObjectMap.keys())
	{
		if(key.contains(customName)) 
			++duplication;
	}
	if (duplication) 
		name += "_" + QString::number(duplication);

	GameObjectPtr go(new GameObject(m_scene, parent));
	go->setObjectName(name);

	registerGameObject(name, go);
	
	return go;
}

void ObjectManager::deleteObject( const QString& name )
{
	GameObjectPtr go = m_gameObjectMap.take(name);
	foreach(ComponentPtr comp, m_renderQueue)
	{
		if (comp->gameObject() == go.data())
		{
			int idx = m_renderQueue.indexOf(comp);
			m_renderQueue.removeAt(idx);
		}
	}
	go.clear();
}

Scene* ObjectManager::getScene() const
{
	return m_scene;
}

void ObjectManager::addComponentToRenderQueue( ComponentPtr comp )
{
	// make sure the components in a smaller render layer gets in to the front
	if (comp->renderLayer() < 0) return;
	
	int target = comp->renderLayer();
	if (m_renderQueue.isEmpty()) 
		m_renderQueue.push_back(comp);
	else
	{
		if (target <= m_renderQueue.front()->renderLayer())
		{
			m_renderQueue.prepend(comp);
		}
		else if (target >= m_renderQueue.last()->renderLayer())
		{
			m_renderQueue.push_back(comp);
		}
		else
		{
			for (int i = 1; i < m_renderQueue.size(); ++i)
			{
				int prev = m_renderQueue[i - 1]->renderLayer();
				int next = m_renderQueue[i]->renderLayer();
				if (target >= prev && target <= next)
				{
					m_renderQueue.insert(i, comp);
					break;
				}
				else
					m_renderQueue.push_back(comp);
			}
		}
	}
}

void ObjectManager::removeComponentFromRenderQueue( ComponentPtr comp )
{
	// make sure the component is renderable
	if (comp->renderLayer() < 0) return;
	int idx = m_renderQueue.indexOf(comp);
	m_renderQueue.remove(idx);
}

void ObjectManager::setLoadingFlag( const QString& flag )
{
	m_loadingFlag = flag;
}

void ObjectManager::renderDebugInfo(const float currentTime)
{
	// display the bounding boxes for each object
	foreach(ComponentPtr comp, m_renderQueue)
	{
		ModelPtr model = comp.dynamicCast<AbstractModel>();
		if (model)
		{
			BoxColliderPtr box = model->getBoundingBox();
			if (box)
				box->render(currentTime);
		}
	}
}
