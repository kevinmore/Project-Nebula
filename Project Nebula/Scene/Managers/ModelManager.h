#pragma once
#include <Scene/RiggedModel.h>
#include <Scene/StaticModel.h>
#include <Utility/ModelLoader.h>
#include <QtCore/QMap>
#include <QtCore/QString>
#include <QtCore/QSharedPointer>

typedef QSharedPointer<AbstractModel> ModelPtr;

class Scene;

class ModelManager
{
public:
	ModelManager(Scene* scene);
	~ModelManager(void);

	ModelPtr getModel(const QString& name);
	ModelPtr loadModel(const QString& name, const QString& filename, ModelLoader::MODEL_TYPE type);
	void renderAllModels(float time);
private:
	QMap<QString, ModelPtr> m_models;
	Scene* m_scene;
};

