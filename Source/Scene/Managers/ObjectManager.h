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

	GameObjectPtr createGameObject(const QString& customName, GameObject* parent = 0);

	void setLoadingFlag(const QString& flag);

	void renderAll(const float currentTime);
	void renderDebugInfo(const float currentTime);


	void deleteObject(const QString& name);

	void clear();

	QMap<QString, GameObjectPtr> m_gameObjectMap;
	QVector<ModelLoaderPtr> m_modelLoaders;
	QVector<ComponentPtr> m_renderQueue;
	QString m_loadingFlag;

public slots:
	void registerComponent(ComponentPtr comp);


private:
	Scene* m_scene;
};

typedef QSharedPointer<ObjectManager> ObjectManagerPtr;
