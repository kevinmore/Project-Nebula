#pragma once
#include <QThread>
#include <Primitives/GameObject.h>
#include <Scene/Managers/ObjectManager.h>

class Scene;
class LoaderThread : public QThread
{
	Q_OBJECT

public:
	LoaderThread(const QString fileName, GameObjectPtr reference = GameObjectPtr(), bool generateGameObject = true);
	~LoaderThread();
	void run();

private:

	ModelPtr loadModel(const QString& customName, const QString& fileName, bool generateGameObject = true);

	ObjectManager* m_objectManager;
	QString m_fileName;
	GameObjectPtr m_reference;
	bool m_shouldGenerateGameObject;

signals:
	void jobDone();
};

