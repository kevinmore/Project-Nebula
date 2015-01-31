#pragma once
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <QObject>

class AssimpThread : public QObject
{
	Q_OBJECT

public slots:
	void readFile(const QString& fileName, uint flags);

signals:
	void resultReady(const aiScene* result);

private:
	Assimp::Importer m_importer;
};

