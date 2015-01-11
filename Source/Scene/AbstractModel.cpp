#include "AbstractModel.h"
AbstractModel::AbstractModel(const QString& fileName, GameObject* go, QObject* parent) 
	: QObject(parent),
	  m_fileName(fileName),
	  m_actor(go)
{}
AbstractModel::~AbstractModel() {}