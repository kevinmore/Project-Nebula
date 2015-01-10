#include "LoaderThread.h"
#include <QFileDialog>
#include <Scene/Scene.h>
LoaderThread::LoaderThread(Scene* scene, QObject* parent)
	: QThread(parent),
	  m_scene(scene)
{
	connect(this, SIGNAL(jobDone()), this, SLOT(quit()));
}


LoaderThread::~LoaderThread()
{}

void LoaderThread::run()
{
	QMutex mutex;
	mutex.lock();

	QString fileName = QFileDialog::getOpenFileName(0, tr("Load Model"),
		"../Resource/Models",
		tr("3D Model File (*.dae *.obj *.3ds)"));

	if (!fileName.isEmpty())
	{
		// extract the file name
		int left = fileName.lastIndexOf("/");
		int right = fileName.lastIndexOf(".");
		QString customName = fileName.mid(left + 1, right - left - 1);

		// extract the relative path
		QDir dir;
		QString relativePath = dir.relativeFilePath(fileName);

		ModelPtr model = m_scene->modelManager()->loadModel(customName, relativePath);
		m_scene->getCamera()->followTarget(model->gameObject());
	}

	mutex.unlock();
	emit jobDone();
}
