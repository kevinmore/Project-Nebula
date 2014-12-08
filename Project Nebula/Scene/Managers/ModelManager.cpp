#include "ModelManager.h"
#include <Scene/Scene.h>
#include <Animation/FK/FKController.h>
#include <Animation/IK/FABRIKSolver.h>
#include <Scene/ShadingTechniques/ShadingTechnique.h>

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

ModelPtr ModelManager::loadModel( const QString& name, const QString& filename, ModelLoader::MODEL_TYPE type )
{

	if (type == ModelLoader::RIGGED_MODEL)
	{

		ShadingTechnique* effect = new ShadingTechnique("../Resource/Shaders/skinning.vert", "../Resource/Shaders/skinning.frag");
		if (!effect->Init()) 
		{
			printf("Error initializing the lighting technique\n");
		}

		ModelLoader* modelLoader = new ModelLoader(effect->getShader()->programId());
		QVector<ModelDataPtr> modelDataVector = modelLoader->loadModel(filename, type);

		// create a FKController for the model
		FKController* controller = new FKController(modelLoader, modelLoader->getSkeletom());

		// create an IKSolver for the model
		CCDIKSolver* solver = new CCDIKSolver(128);

		RiggedModel* rm = new RiggedModel(m_scene, effect, modelLoader->getSkeletom(), controller, solver, modelLoader->getVAO(), modelDataVector);
		m_riggedModels[name] = rm;
		m_allModels[name] = ModelPtr(rm);
	}

	else if (type == ModelLoader::STATIC_MODEL)
	{
		ShadingTechnique* effect = new ShadingTechnique("../Resource/Shaders/static.vert", "../Resource/Shaders/static.frag");
		if (!effect->Init()) 
		{
			printf("Error initializing the lighting technique\n");
		}

		ModelLoader* modelLoader = new ModelLoader(effect->getShader()->programId());
		QVector<ModelDataPtr> modelDataVector = modelLoader->loadModel(filename, type);

		StaticModel* sm = new StaticModel(m_scene, effect, modelLoader->getVAO(), modelDataVector);
		m_staticModels[name] = sm;
		m_allModels[name] = ModelPtr(sm);

	}

	return m_allModels[name];

}

void ModelManager::renderAllModels(float time)
{
	QMap<QString, ModelPtr>::Iterator i;

	for (i = m_allModels.begin(); i != m_allModels.end(); ++i)
	{
		i.value()->render(time);
	}

}

void ModelManager::renderRiggedModels( float time )
{
	QMap<QString, RiggedModel*>::Iterator i;

	for (i = m_riggedModels.begin(); i != m_riggedModels.end(); ++i)
	{
		i.value()->render(time);
	}
}

void ModelManager::renderStaticModels( float time )
{
	QMap<QString, StaticModel*>::Iterator i;

	for (i = m_staticModels.begin(); i != m_staticModels.end(); ++i)
	{
		i.value()->render(time);
	}
}
