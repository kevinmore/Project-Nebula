#pragma once
#include <Scene/RiggedModel.h>
#include <Scene/StaticModel.h>
#include <Utility/ModelLoader.h>

#include <Utility/EngineCommon.h>
typedef QSharedPointer<GameObject> GameObjectPtr;
typedef QSharedPointer<ModelLoader> ModelLoaderPtr;

class Scene;

class ObjectManager
{
public:
	ObjectManager(Scene* scene);
	~ObjectManager();

	GameObjectPtr getGameObject(const QString& name);
	ModelPtr getModel(const QString& name);

	GameObjectPtr createGameObject(const QString& customName, GameObject* parent = 0);
	ModelPtr loadModel(const QString& customName, const QString& fileName, GameObject* parent = 0);
	void renderAllModels(float time);

	void deleteObject(const QString& name);

	void clear();
	void gatherModelsInfo();

	QMap<QString, GameObjectPtr> m_gameObjectMap;

	QVector<ModelLoaderPtr> m_modelLoaders;
	QVector<QPair<QString, GameObject*>> m_modelsInfo;

private:
	Scene* m_scene;
};

