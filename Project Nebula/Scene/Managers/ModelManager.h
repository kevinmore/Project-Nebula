#pragma once
#include <Scene/RiggedModel.h>
#include <Scene/StaticModel.h>
#include <Utility/ModelLoader.h>
#include <QMap>
#include <QString>
#include <QSharedPointer>

typedef QSharedPointer<AbstractModel> ModelPtr;

class Scene;

class ModelManager
{
public:
	ModelManager(Scene* scene);
	~ModelManager(void);

	ModelPtr getModel(const QString& name);
	RiggedModel* getRiggedModel(const QString& name);
	StaticModel* getStaticModel(const QString& name);

	ModelPtr loadModel(const QString& name, const QString& filename, ModelLoader::MODEL_TYPE type);
	void renderAllModels(float time);
	void renderRiggedModels(float time);
	void renderStaticModels(float time);

	QMap<QString, ModelPtr> m_allModels;
	QMap<QString, RiggedModel*> m_riggedModels;
	QMap<QString, StaticModel*> m_staticModels;

private:
	Scene* m_scene;
};

