#pragma once
#include <QThread>
#include <Scene/GameObject.h>
class Scene;
class LoaderThread : public QThread
{
	Q_OBJECT

public:
	LoaderThread(Scene* scene, const QString fileName, GameObject* reference = 0, GameObject* objectParent = 0);
	~LoaderThread();
	void run();

private:
	Scene* m_scene;
	QString m_fileName;
	GameObject* m_reference;
	GameObject* m_objectParent;

signals:
	void jobDone();

public slots:

};

