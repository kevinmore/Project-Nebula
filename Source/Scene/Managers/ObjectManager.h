#pragma once
#include <Scene/RiggedModel.h>
#include <Scene/StaticModel.h>
#include <Utility/ModelLoader.h>

#include <Utility/EngineCommon.h>
typedef QSharedPointer<ModelLoader> ModelLoaderPtr;

class Scene;

class ObjectManager
{
public:
	ObjectManager(Scene* scene);
	~ObjectManager();

	Scene* getScene() const;
	void registerGameObject(const QString& name, GameObjectPtr go);
	GameObjectPtr getGameObject(const QString& name);
	ModelPtr getModel(const QString& name);

	GameObjectPtr createGameObject(const QString& customName, GameObject* parent = 0);

	ModelPtr loadModel(const QString& customName, const QString& fileName, 
						GameObject* parent = 0, bool generateGameObject = true);

	void renderAll(const float currentTime);

	void deleteObject(const QString& name);

	void clear();

	QMap<QString, GameObjectPtr> m_gameObjectMap;
	QVector<ModelLoaderPtr> m_modelLoaders;
private:
	Scene* m_scene;
};

typedef QSharedPointer<ObjectManager> ObjectManagerPtr;
