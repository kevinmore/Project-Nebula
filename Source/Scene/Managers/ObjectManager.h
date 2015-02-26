#pragma once
#include <QObject>
#include <Scene/RiggedModel.h>
#include <Scene/StaticModel.h>
#include <Utility/ModelLoader.h>
#include <Utility/EngineCommon.h>

class Scene;

class ObjectManager : QObject
{
	Q_OBJECT

public:
	ObjectManager(Scene* scene, QObject* parent = 0);
	~ObjectManager();

	Scene* getScene() const;
	void registerGameObject(const QString& name, GameObjectPtr go);

	GameObjectPtr getGameObject(const QString& name);

	// returns a list containing all the game objects, from parent to child order
	QList<GameObjectPtr> getGameObjectTreeList();

	GameObjectPtr createGameObject(const QString& customName, GameObject* parent = 0);

	void setLoadingFlag(const QString& flag);

	void renderAll(const float currentTime);

	void deleteObject(const QString& name);

	void clear();

	QMap<QString, GameObjectPtr> m_gameObjectMap;
	QVector<ModelLoaderPtr> m_modelLoaders;
	QVector<ComponentPtr> m_renderQueue;
	QString m_loadingFlag;
	QList<GameObject*> m_gameObjectTreeList;

public slots:
	void addComponentToRenderQueue(ComponentPtr comp);
	void removeComponentFromRenderQueue(ComponentPtr comp);

private:
	void readHierarchy(GameObject* root);

	Scene* m_scene;
};

typedef QSharedPointer<ObjectManager> ObjectManagerPtr;
