#pragma once
#include <QThread>
#include <Scene/GameObject.h>
class Scene;
class LoaderThread : public QThread
{
	Q_OBJECT

public:
	LoaderThread(Scene* scene, const QString fileName, GameObject* go = 0, QObject* parent = 0);
	~LoaderThread();
	void run();

private:
	Scene* m_scene;
	QString m_fileName;
	GameObject* m_actor;

signals:
	void jobDone();

public slots:

};

