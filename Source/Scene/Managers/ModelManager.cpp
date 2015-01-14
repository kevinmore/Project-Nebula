#include "ModelManager.h"
#include <Scene/Scene.h>
#include <Animation/FK/FKController.h>
#include <Animation/IK/FABRIKSolver.h>

ModelManager::ModelManager(Scene* scene)
	: m_scene(scene)
{}


ModelManager::~ModelManager() {}

GameObject* ModelManager::getGameObject( const QString& name )
{
	if(m_gameObjectMap.find(name) != m_gameObjectMap.end()) return m_gameObjectMap[name];
	else return 0;
}

ModelPtr ModelManager::getModel( const QString& name )
{
	if(m_modelMap.find(name) != m_modelMap.end()) return m_modelMap[name];
	else return ModelPtr();
}

ModelPtr ModelManager::loadModel( const QString& customName, const QString& fileName, GameObject* parent )
{
	ModelLoader* m_modelLoader = new ModelLoader();
	QVector<ModelDataPtr> modelDataArray = m_modelLoader->loadModel(fileName);
	if(modelDataArray.size() == 0) return ModelPtr();

	// if the model has mesh data, load it
	ModelPtr pModel;

	// check if this model has been loaded previously
	QString name = customName;
	int duplication = 0;
	foreach(QString key, m_modelMap.keys())
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

	// link this model to a new game object
	GameObject* go  = new GameObject(parent);
	pModel->linkGameObject(go);
	pModel->gameObject()->setParent(parent);
	pModel->gameObject()->setObjectName(name);

	// add the data into the maps
	m_gameObjectMap[name] = pModel->gameObject();
	m_modelMap[name] = pModel;
	m_modelLoaders.push_back(m_modelLoader);

	return pModel;
}

void ModelManager::renderAllModels(float time)
{
	foreach(ModelPtr model, m_modelMap.values())
	{
		model->render(time);
	}
}

void ModelManager::clear()
{
	for (int i = 0; i < m_modelLoaders.size(); ++i)
	{
		SAFE_DELETE(m_modelLoaders[i]);
	}

	foreach(ModelPtr model, m_modelMap.values())
	{
		model.clear();
	}

	m_modelLoaders.clear();
	m_modelMap.clear();
	m_gameObjectMap.clear();
}

void ModelManager::gatherModelsInfo()
{
	m_modelsInfo.clear();
	foreach(ModelPtr model, m_modelMap.values())
	{
		m_modelsInfo.push_back(qMakePair(model->fileName(), model->gameObject()));
	}
}

GameObject* ModelManager::createGameObject( const QString& customName, GameObject* parent /*= 0*/ )
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

	GameObject* go = new GameObject(parent);
	go->setObjectName(name);
	m_gameObjectMap[name] = go;

	return go;
}

void ModelManager::deleteObject( const QString& name )
{
	if(getGameObject(name)) 
		m_gameObjectMap.take(name);
	if(getModel(name)) 
		m_modelMap.take(name);
}
