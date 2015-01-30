#include "ObjectManager.h"
#include <Scene/Scene.h>
#include <Animation/FK/FKController.h>
#include <Animation/IK/FABRIKSolver.h>

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
		registerComponent(comp);
	}

	// listen for further updates
	connect(go.data(), SIGNAL(componentAttached(ComponentPtr)), this, SLOT(registerComponent(ComponentPtr)));
}

GameObjectPtr ObjectManager::getGameObject( const QString& name )
{
	if(m_gameObjectMap.find(name) != m_gameObjectMap.end()) return m_gameObjectMap[name];
	else return GameObjectPtr();
}

ModelPtr ObjectManager::loadModel( const QString& customName, const QString& fileName, GameObject* parent, bool generateGameObject )
{
	ModelPtr pModel;

	// check if the custom name is unique
	QString name = customName;
	int duplication = 0;
	foreach(QString key, m_gameObjectMap.keys())
	{
		if(key.contains(customName)) 
			++duplication;
	}
	if (duplication) 
		name += "_" + QString::number(duplication);

	// if the model already exists, make a copy
	foreach(ComponentPtr comp, m_renderQueue)
	{
		ModelPtr model = comp.dynamicCast<AbstractModel>();
		if (model && fileName == model->fileName())
		{
			pModel = model;
			break;
		}
	}
	if (pModel)
	{
		// check model type
		if (pModel.dynamicCast<StaticModel>())
		{
			StaticModel* original = dynamic_cast<StaticModel*>(pModel.data());
			StaticModel* copyModel = new StaticModel(original);
			pModel.reset(copyModel);
		}
		else if (pModel.dynamicCast<RiggedModel>())
		{
			RiggedModel* original = dynamic_cast<RiggedModel*>(pModel.data());
			RiggedModel* copyModel = new RiggedModel(original);
			pModel.reset(copyModel);
		}
		qDebug() << "Made a copy from" << fileName;
	}
	// if the model doesn't exist, load it from file
	else
	{
		ModelLoaderPtr modelLoader(new ModelLoader(m_scene));
		QVector<ModelDataPtr> modelDataArray = modelLoader->loadModel(fileName, 0, m_loadingFlag);
		if(modelDataArray.size() == 0) return pModel;

		// create different types of models
		if (modelLoader->getModelType() == ModelLoader::STATIC_MODEL)
		{
			StaticModel* sm = new StaticModel(fileName, m_scene, modelLoader->getRenderingEffect(), modelDataArray);
			pModel.reset(sm);
		}
		else if (modelLoader->getModelType() == ModelLoader::RIGGED_MODEL)
		{
			RiggedModel* rm = new RiggedModel(fileName, m_scene, modelLoader, modelDataArray);
			pModel.reset(rm);
		}
		m_modelLoaders.push_back(modelLoader);
	}
	
	if (generateGameObject)
	{
		// attach this model to a new game object
		GameObjectPtr go(new GameObject(m_scene, parent));
		go->setObjectName(name);
		go->attachComponent(pModel);

		// add the data into the maps
		registerGameObject(name, go);
	}

	return pModel;
}

void ObjectManager::renderAll(const float currentTime)
{
	// sync physics engine feedback
// 	foreach(GameObjectPtr go, m_gameObjectMap.values())
// 	{
// 		RigidBodyPtr rb = go->getComponent("RigidBody").dynamicCast<RigidBody>();
// 		if (rb)
// 		{
// 			QObjectList children = go->children();
// 			foreach(QObject* obj, children)
// 			{
// 				GameObject* child = dynamic_cast<GameObject*>(obj);
// 				ParticleSystemPtr ps = child->getComponent("ParticleSystem").dynamicCast<ParticleSystem>();
// 				rb->applyPointImpulse(ps->getLinearImpuse() * 0.8f, child->position() / 100.0f);
// 			}
// 			rb->applyAngularImpulse(vec3(0,0.1,0));
// 			go->setTransformMatrix(rb->getTransformMatrix());
// 		}
// 	}

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

void ObjectManager::registerComponent( ComponentPtr comp )
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

void ObjectManager::setLoadingFlag( const QString& flag )
{
	m_loadingFlag = flag;
}
