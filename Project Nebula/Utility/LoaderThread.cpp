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
		ModelPtr model = m_scene->modelManager()->loadModel(fileName, fileName);
		m_scene->getCamera()->followTarget(model->getActor());
	}

	mutex.unlock();
	emit jobDone();
}
