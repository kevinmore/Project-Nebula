#pragma once
#include <QThread>
#include <Primitives/GameObject.h>
#include <Scene/Managers/ObjectManager.h>

class Scene;
class LoaderThread : public QThread
{
	Q_OBJECT

public:
	LoaderThread(Scene* scene, const QString fileName, GameObjectPtr reference = GameObjectPtr(), 
				GameObject* objectParent = 0, bool generateGameObject = true);
	~LoaderThread();
	void run();

private:

	ModelPtr loadModel(const QString& customName, const QString& fileName, 
		GameObject* parent = 0, bool generateGameObject = true);

	Scene* m_scene;
	ObjectManager* m_objectManager;
	QString m_fileName;
	GameObjectPtr m_reference;
	GameObject* m_objectParent;
	bool m_shouldGenerateGameObject;

signals:
	void jobDone();

public slots:

};

