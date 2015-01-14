#include "ObjectManager.h"
#include <Scene/Scene.h>
#include <Animation/FK/FKController.h>
#include <Animation/IK/FABRIKSolver.h>

ObjectManager::ObjectManager(Scene* scene)
	: m_scene(scene)
{}


ObjectManager::~ObjectManager() {}

GameObjectPtr ObjectManager::getGameObject( const QString& name )
{
	if(m_gameObjectMap.find(name) != m_gameObjectMap.end()) return m_gameObjectMap[name];
	else return GameObjectPtr();
}

ModelPtr ObjectManager::getModel( const QString& name )
{
	if(m_gameObjectMap.find(name) != m_gameObjectMap.end()) return m_gameObjectMap[name]->getModel();
	else return ModelPtr();
}

ModelPtr ObjectManager::loadModel( const QString& customName, const QString& fileName, GameObject* parent )
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
		FKController* controller = new FKController(m_modelLoader.data(), m_modelLoader->getSkeletom());

		// create an IKSolver for the model
		CCDIKSolver* solver = new CCDIKSolver(128);

		RiggedModel* rm = new RiggedModel(fileName, m_scene, m_modelLoader->getRenderingEffect(), m_modelLoader->getSkeletom(), modelDataArray);
		rm->setFKController(controller);
		rm->setIKSolver(solver);
		rm->setRootTranslation(controller->getRootTranslation());
		rm->setRootRotation(controller->getRootRotation());
		pModel.reset(rm);
	}

	// attach this model to a new game object
	GameObjectPtr go(new GameObject(parent));
	go->setObjectName(name);
	go->attachModel(pModel);

	// add the data into the maps
	m_gameObjectMap[name] = go;
	m_modelLoaders.push_back(m_modelLoader);

	return pModel;
}

void ObjectManager::renderAllModels(float time)
{
	foreach(GameObjectPtr go, m_gameObjectMap.values())
	{
		ModelPtr model = go->getModel();
		if (!model.isNull())
		{
			model->render(time);
		}
	}
}

void ObjectManager::clear()
{
	// clean up
	// since we are using QSharedPointer here, there's no need to destroy each object individually
	m_modelLoaders.clear();
	m_gameObjectMap.clear();
}

void ObjectManager::gatherModelsInfo()
{
	m_modelsInfo.clear();
	foreach(GameObjectPtr go, m_gameObjectMap.values())
	{
		m_modelsInfo.push_back(qMakePair(go->getModel()->fileName(), go.data()));
	}
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

	GameObjectPtr go(new GameObject(parent));
	go->setObjectName(name);
	m_gameObjectMap[name] = go;

	return go;
}

void ObjectManager::deleteObject( const QString& name )
{
	if(getGameObject(name)) 
		m_gameObjectMap.take(name);
}