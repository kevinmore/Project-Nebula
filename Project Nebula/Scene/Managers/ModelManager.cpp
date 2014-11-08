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
	if(m_models.find(name) != m_models.end()) return m_models[name];
	else return ModelPtr();
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
		FKController* controller = new FKController(modelLoader);

		// create an IKSolver for the model
		CCDIKSolver* solver = new CCDIKSolver();

		m_models[name] = ModelPtr(new RiggedModel(m_scene, effect, modelLoader->getSkeletom(), controller, solver, modelLoader->getVAO(), modelDataVector));

	}

	else if (type == ModelLoader::STATIC_MODEL)
	{
		ShadingTechnique* effect = new ShadingTechnique("../Resource/Shaders/skinning.vert", "../Resource/Shaders/skinning.frag");
		if (!effect->Init()) 
		{
			printf("Error initializing the lighting technique\n");
		}

		ModelLoader* modelLoader = new ModelLoader(effect->getShader()->programId());
		QVector<ModelDataPtr> modelDataVector = modelLoader->loadModel(filename, type);

		m_models[name] = ModelPtr(new StaticModel(m_scene, effect, modelLoader->getVAO(), modelDataVector));

	}

	return m_models[name];

}

void ModelManager::renderAllModels(float time)
{
	QMap<QString, ModelPtr>::Iterator i;

	for (i = m_models.begin(); i != m_models.end(); ++i)
	{
		i.value()->render(time);
	}

}
