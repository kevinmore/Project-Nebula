#pragma once
#include <QThread>
class Scene;
class LoaderThread : public QThread
{
	Q_OBJECT

public:
	LoaderThread(Scene* scene, QObject* parent = 0);
	~LoaderThread();
	void run();

private:
	Scene* m_scene;

signals:
	void jobDone();

public slots:

};

