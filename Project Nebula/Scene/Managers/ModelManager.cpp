#include "ModelManager.h"
#include <Scene/Scene.h>

ModelManager::ModelManager(Scene* scene)
	: m_scene(scene),
	m_modelLoader(new ModelLoader())
{}


ModelManager::~ModelManager() {}

ModelPtr ModelManager::getModel( const QString& name )
{
	if(m_models.find(name) != m_models.end()) return m_models[name];
	else return ModelPtr();
}

ModelPtr ModelManager::loadModel( const QString& name, const QString& filename, const QOpenGLShaderProgramPtr& shaderProgram )
{
	QVector<ModelDataPtr> modelDataVector = m_modelLoader->loadModel(filename, shaderProgram);
	m_models[name] = ModelPtr(new Model(m_scene, m_modelLoader, m_modelLoader->getVAO(), modelDataVector));

	return m_models[name];
}
