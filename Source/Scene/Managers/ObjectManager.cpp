#include "ObjectManager.h"
#include <Scene/Scene.h>
#include <Animation/FK/FKController.h>
#include <Animation/IK/FABRIKSolver.h>

ObjectManager::ObjectManager(Scene* scene, QObject* parent)
	: QObject(parent),
	  m_scene(scene)
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
	ModelLoaderPtr m_modelLoader(new ModelLoader);
	QVector<ModelDataPtr> modelDataArray = m_modelLoader->loadModel(fileName);
	if(modelDataArray.size() == 0) return ModelPtr();

	// if the model has mesh data, load it
	ModelPtr pModel;

	// check if this model has been loaded previously
	QString name = customName;
	int duplication = 0;
	foreach(QString key, m_gameObjectMap.keys())
	{
		if(key.contains(customName)) 
			++duplication;
	}
	if (duplication) 
		name += "_" + QString::number(duplication);

	// create different types of models
	if (m_modelLoader->getModelType() == ModelLoader::STATIC_MODEL)
	{
		StaticModel* sm = new StaticModel(fileName, m_scene, m_modelLoader->getRenderingEffect(), modelDataArray);
		pModel.reset(sm);
	}
	else if (m_modelLoader->getModelType() == ModelLoader::RIGGED_MODEL)
	{
		// create a FKController for the model
		FKController* controller = new FKController(m_modelLoader, m_modelLoader->getSkeletom());

		// create an IKSolver for the model
		CCDIKSolver* solver = new CCDIKSolver(128);

		RiggedModel* rm = new RiggedModel(fileName, m_scene, m_modelLoader->getRenderingEffect(), m_modelLoader->getSkeletom(), modelDataArray);
		rm->setFKController(controller);
		rm->setIKSolver(solver);
		rm->setRootTranslation(controller->getRootTranslation());
		rm->setRootRotation(controller->getRootRotation());
		pModel.reset(rm);
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

	m_modelLoaders.push_back(m_modelLoader);
	return pModel;
}

void ObjectManager::renderAll(const float currentTime)
{
	// sync physics engine feedback
	foreach(GameObjectPtr go, m_gameObjectMap.values())
	{
		RigidBodyPtr rb = go->getComponent("RigidBody").dynamicCast<RigidBody>();
		if (rb)
		{
			
			rb->applyAngularImpulse(vec3(0,0.01,0));
			vec3 angle = rb->getRotationInAxisAndAngles();
			go->setPosition(rb->getPosition());
			go->setRotation(angle);
			//go->translateInWorld(rb->getDeltaPosition());
			//go->rotateInWorld(rb->getDeltaRotation());
			//quart q = quart::fromAxisAndAngle(vec3(0, 1, 0), 2);
			//go->rotateInWorld(q);
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
	// since we are using QSharedPointer here, there's no need to destroy each object individually
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
	if(getGameObject(name)) 
		m_gameObjectMap.take(name);
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
