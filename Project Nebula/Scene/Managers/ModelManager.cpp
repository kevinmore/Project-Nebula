#include "ModelManager.h"
#include <Scene/Scene.h>

ModelManager::ModelManager(Scene* scene)
	: m_scene(scene),
	m_modelLoader(ModelLoader())
{}


ModelManager::~ModelManager(void)
{
}

ModelPtr ModelManager::getModel( const QString& name )
{
	if(m_models.find(name) != m_models.end()) return m_models[name];
	else return ModelPtr();
}

ModelPtr ModelManager::loadModel( const QString& name, const QString& filename, const QOpenGLShaderProgramPtr& shaderProgram )
{
	QVector<ModelDataPtr> modelDataVector = m_modelLoader.loadModel(filename, shaderProgram);
	Model* md = new Model(m_scene, m_modelLoader.getVAO(), modelDataVector);
	m_models[name] = ModelPtr(md);

	delete md;
	md = nullptr;

	return m_models[name];
}
