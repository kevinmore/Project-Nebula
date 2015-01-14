#pragma once
#include <Scene/RiggedModel.h>
#include <Scene/StaticModel.h>
#include <Utility/ModelLoader.h>

#include <Utility/EngineCommon.h>
typedef QSharedPointer<AbstractModel> ModelPtr;

class Scene;

class ModelManager
{
public:
	ModelManager(Scene* scene);
	~ModelManager(void);

	GameObject* getGameObject(const QString& name);
	ModelPtr getModel(const QString& name);

	GameObject* createGameObject(const QString& customName, GameObject* parent = 0);
	ModelPtr loadModel(const QString& customName, const QString& fileName, GameObject* parent = 0);
	void renderAllModels(float time);

	void deleteObject(const QString& name);

	void clear();
	void gatherModelsInfo();

	QMap<QString, GameObject*> m_gameObjects;
	QMap<QString, ModelPtr> m_allModels;

	QVector<ModelLoader*> m_modelLoaders;
	QVector<QPair<QString, GameObject*>> m_modelsInfo;

private:
	Scene* m_scene;
};

