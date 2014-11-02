#include "ModelManager.h"
#include <Scene/Scene.h>
#include <Animation/FK/FKController.h>

ModelManager::ModelManager(Scene* scene)
	: m_scene(scene)
{}


ModelManager::~ModelManager() {}

ModelPtr ModelManager::getModel( const QString& name )
{
	if(m_models.find(name) != m_models.end()) return m_models[name];
	else return ModelPtr();
}

ModelPtr ModelManager::loadModel( const QString& name, const QString& filename )
{
	ModelLoader* modelLoader = new ModelLoader();
	QVector<ModelDataPtr> modelDataVector = modelLoader->loadModel(filename);

	// create a FKController for the model
	FKController* controller = new FKController(modelLoader);

	m_models[name] = ModelPtr(new Model(m_scene, controller, QOpenGLVertexArrayObjectPtr(modelLoader->getVAO()), modelDataVector));
	//delete modelLoader;
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
