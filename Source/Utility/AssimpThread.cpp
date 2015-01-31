#include "AssimpThread.h"

void AssimpThread::readFile( const QString& fileName, uint flags )
{
	const aiScene* scene = m_importer.ReadFile(fileName.toStdString(), flags);

	emit resultReady(scene);
}
