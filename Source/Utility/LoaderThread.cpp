#include "LoaderThread.h"
#include <QFileDialog>
#include <Scene/Scene.h>
LoaderThread::LoaderThread(Scene* scene, const QString fileName, GameObjectPtr reference, GameObject* objectParent, bool generateGameObject)
	: QThread(scene),
	  m_scene(scene),
	  m_fileName(fileName),
	  m_reference(reference),
	  m_objectParent(objectParent),
	  m_shouldGenerateGameObject(generateGameObject)
{
	connect(this, SIGNAL(jobDone()), m_scene, SLOT(modelLoaded()));
	run();
}

LoaderThread::~LoaderThread()
{
}

void LoaderThread::run()
{
	QMutex mutex;
	mutex.lock();

	if (!m_fileName.isEmpty())
	{
		// extract the file name
		int left = m_fileName.lastIndexOf("/");
		int right = m_fileName.lastIndexOf(".");
		QString customName = m_fileName.mid(left + 1, right - left - 1);

		// extract the relative path
		QDir dir;
		QString relativePath = dir.relativeFilePath(m_fileName);

		ModelPtr model = m_scene->objectManager()->loadModel(customName, relativePath, m_objectParent, m_shouldGenerateGameObject);
		// if not generating a game object for the model
		// let it attach to its reference object
		if (!m_shouldGenerateGameObject) m_reference->attachComponent(model);

		// apply transformation to this model
		if (m_reference)
		{
			model->gameObject()->setPosition(m_reference->position());
			model->gameObject()->setRotation(m_reference->rotation());
			model->gameObject()->setScale(m_reference->scale());
		}
	}

	mutex.unlock();

	// emit the signal and destroy the thread
	emit jobDone();
	msleep(1000);
	quit();
}
