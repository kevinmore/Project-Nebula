#include "ModelManager.h"
#include <Scene/Scene.h>
#include <Animation/FK/FKController.h>
#include <Animation/IK/FABRIKSolver.h>

ModelManager::ModelManager(Scene* scene)
	: m_scene(scene)
{}


ModelManager::~ModelManager() {}

ModelPtr ModelManager::getModel( const QString& name )
{
	if(m_allModels.find(name) != m_allModels.end()) return m_allModels[name];
	else return ModelPtr();
}

RiggedModel* ModelManager::getRiggedModel( const QString& name )
{
	if(m_riggedModels.find(name) != m_riggedModels.end()) return m_riggedModels[name];
	else return NULL;
}

StaticModel* ModelManager::getStaticModel( const QString& name )
{
	if(m_staticModels.find(name) != m_staticModels.end()) return m_staticModels[name];
	else return NULL;
}

ModelPtr ModelManager::loadModel( const QString& name, const QString& filename )
{
	ModelLoader* m_modelLoader = new ModelLoader();
	QVector<ModelDataPtr> modelDataArray = m_modelLoader->loadModel(filename);
	if(modelDataArray.size() == 0) return ModelPtr();

	if (m_modelLoader->getModelType() == ModelLoader::STATIC_MODEL)
	{
		StaticModel* sm = new StaticModel(m_scene, m_modelLoader->getRenderingEffect(), modelDataArray);
		m_staticModels[name] = sm;
		m_allModels[name] = ModelPtr(sm);
	}
	else if (m_modelLoader->getModelType() == ModelLoader::RIGGED_MODEL)
	{
		// create a FKController for the model
		FKController* controller = new FKController(m_modelLoader, m_modelLoader->getSkeletom());

		// create an IKSolver for the model
		CCDIKSolver* solver = new CCDIKSolver(128);

		RiggedModel* rm = new RiggedModel(m_scene, m_modelLoader->getRenderingEffect(), m_modelLoader->getSkeletom(), controller, solver, modelDataArray);
		rm->setRootTranslation(controller->getRootTranslation());
		rm->setRootRotation(controller->getRootRotation());
		m_riggedModels[name] = rm;
		m_allModels[name] = ModelPtr(rm);
	}

	m_modelLoaders.push_back(m_modelLoader);
	return m_allModels[name];
}

void ModelManager::renderAllModels(float time)
{
	foreach(ModelPtr model, m_allModels.values())
	{
		model->render(time);
	}
}

void ModelManager::renderRiggedModels( float time )
{
	foreach(RiggedModel* model, m_riggedModels.values())
	{
		model->render(time);
	}
}

void ModelManager::renderStaticModels( float time )
{
	foreach(StaticModel* model, m_staticModels.values())
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

	for (auto it = m_allModels.begin(); it != m_allModels.end(); )
	{
		m_allModels.erase(it++);
	}

	m_modelLoaders.clear();
	m_staticModels.clear();
	m_riggedModels.clear();
	m_allModels.clear();
}
