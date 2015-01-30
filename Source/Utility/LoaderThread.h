#pragma once
#include <QThread>
#include <Primitives/GameObject.h>
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
	Scene* m_scene;
	QString m_fileName;
	GameObjectPtr m_reference;
	GameObject* m_objectParent;
	bool m_shouldGenerateGameObject;

signals:
	void jobDone();

public slots:

};

