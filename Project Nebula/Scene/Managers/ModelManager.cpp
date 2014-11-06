#include "ModelManager.h"
#include <Scene/Scene.h>
#include <Animation/FK/FKController.h>
#include <Animation/IK/IKSolver.h>

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
	ModelLoader* modelLoader = new ModelLoader();
	QVector<ModelDataPtr> modelDataVector = modelLoader->loadModel(filename, type);

	if (type == ModelLoader::RIGGED_MODEL)
	{
		// create a FKController for the model
		FKController* controller = new FKController(modelLoader);

		// create an IKSolver for the model
		IKSolver* solver = new IKSolver(modelLoader->getSkeletom(), 0.00001f);

		m_models[name] = ModelPtr(new RiggedModel(m_scene, controller, solver, QOpenGLVertexArrayObjectPtr(modelLoader->getVAO()), modelDataVector));

	}

	else if (type == ModelLoader::STATIC_MODEL)
	{
		m_models[name] = ModelPtr(new StaticModel(m_scene, QOpenGLVertexArrayObjectPtr(modelLoader->getVAO()), modelDataVector));

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
